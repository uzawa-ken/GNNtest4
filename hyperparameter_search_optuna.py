"""簡易的な Optuna によるハイパーパラメータ探索スクリプト。

最終検証誤差（相対誤差）の最小値を目的関数とし、
`GNN_train_val_weight.py` のハイパーパラメータを自動調整します。

主な特徴:
- 学習率、Weight Decay、損失の重み（LAMBDA_DATA / LAMBDA_PDE）、GNN の隠れチャネル数 / 層数を探索
- 学習曲線の描画は無効化して高速化
- 返却される検証誤差の最小値を Optuna が最小化
- 乱数シードと train/val 分割比率を引数で指定して再現性を確保

実行例:
    python hyperparameter_search_optuna.py --trials 20 --data_dir ./data

注意:
- Optuna がインストールされていない場合は `pip install optuna` を実行してください。
- 本スクリプトは 1 試行につき `GNN_train_val_weight.py` と同じ学習を行うため、
  試行回数を増やすと計算コストが増えます。少ない試行から始めてください。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np

try:
    import optuna
except ImportError as exc:  # pragma: no cover - ユーザー環境でのみ発生しうるため
    raise RuntimeError(
        "Optuna が見つかりません。`pip install optuna` を実行してください。"
    ) from exc

import GNN_train_val_weight as gnn


def _set_global_params(
    *,
    lr: float,
    weight_decay: float,
    lambda_data: float,
    lambda_pde: float,
    lambda_gauge: float,
    hidden_channels: int,
    num_layers: int,
    num_epochs: int,
    train_fraction: float,
    max_num_cases: int,
    random_seed: int,
) -> None:
    """GNN トレーニングスクリプトのグローバル設定を上書きするヘルパー。"""

    gnn.LR = lr
    gnn.WEIGHT_DECAY = weight_decay
    gnn.LAMBDA_DATA = lambda_data
    gnn.LAMBDA_PDE = lambda_pde
    gnn.LAMBDA_GAUGE = lambda_gauge
    gnn.HIDDEN_CHANNELS = hidden_channels
    gnn.NUM_LAYERS = num_layers
    gnn.NUM_EPOCHS = num_epochs
    gnn.TRAIN_FRACTION = train_fraction
    gnn.MAX_NUM_CASES = max_num_cases
    gnn.RANDOM_SEED = random_seed


def _initialize_log_file(log_file: Path) -> None:
    """ハイパーパラメータ探索ログ用のテキストファイルを作成する。"""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# trial\tval_error\tlr\tweight_decay\tlambda_data\t"
        "lambda_pde\thidden_channels\tnum_layers\n"
    )

    if not log_file.exists():
        log_file.write_text(header, encoding="utf-8")


def _append_trial_result(
    log_file: Path,
    trial_number: int,
    val_error: float,
    *,
    lr: float,
    weight_decay: float,
    lambda_data: float,
    lambda_pde: float,
    hidden_channels: int,
    num_layers: int,
) -> None:
    """試行結果をテキストファイルへ逐次追記する。"""

    _initialize_log_file(log_file)
    line = (
        f"{trial_number}\t{val_error:.6e}\t{lr:.6e}\t{weight_decay:.6e}\t"
        f"{lambda_data:.6e}\t{lambda_pde:.6e}\t{hidden_channels}\t{num_layers}\n"
    )

    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)


def _extract_best_val_error(history: dict) -> float:
    """学習履歴から最良の検証相対誤差を取り出す。"""

    val_errors: List[float] = [v for v in history["rel_err_val"] if v is not None]
    if val_errors:
        return float(np.min(val_errors))

    # 検証データが無い場合は訓練誤差を代用
    if history["rel_err_train"]:
        return float(history["rel_err_train"][-1])

    raise RuntimeError("学習履歴が空のため評価指標を取得できませんでした。")


def objective(
    trial: optuna.Trial,
    data_dir: str,
    num_epochs: int,
    max_num_cases: int,
    train_fraction: float,
    random_seed: int,
    log_file: Path,
    lambda_gauge: float,
    search_lambda_gauge: bool,
) -> float:
    """Optuna 用の目的関数。"""

    # サンプルするハイパーパラメータ
    lr = trial.suggest_float(name="lr", low=1e-4, high=1e-2, log=True)
    weight_decay = trial.suggest_float(name="weight_decay", low=1e-6, high=1e-3, log=True)
    lambda_data = trial.suggest_float(name="lambda_data", low=1e-3, high=1.0, log=True)
    lambda_pde = trial.suggest_float(name="lambda_pde", low=1e-3, high=1.0, log=True)
    hidden_channels = trial.suggest_int("hidden_channels", 32, 256, log=True)
    num_layers = trial.suggest_int("num_layers", 3, 7)

    # ゲージ正則化係数（教師なし学習時の定数モード抑制用）
    if search_lambda_gauge:
        lambda_gauge_val = trial.suggest_float(
            name="lambda_gauge", low=1e-4, high=1.0, log=True
        )
    else:
        lambda_gauge_val = lambda_gauge

    _set_global_params(
        lr=lr,
        weight_decay=weight_decay,
        lambda_data=lambda_data,
        lambda_pde=lambda_pde,
        lambda_gauge=lambda_gauge_val,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_epochs=num_epochs,
        train_fraction=train_fraction,
        max_num_cases=max_num_cases,
        random_seed=random_seed,
    )

    # 乱数シードをそろえて再現性を確保
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 学習を実行（プロットなし）
    history = gnn.train_gnn_auto_trainval_pde_weighted(
        data_dir,
        enable_plot=False,          # 探索中はリアルタイムプロットもオフ
        return_history=True,
        enable_error_plots=False,   # ★ 探索中は誤差場プロットも完全オフ
    )

    # 目的関数として最小検証相対誤差を返す
    val_error = _extract_best_val_error(history)
    _append_trial_result(
        log_file,
        trial.number,
        val_error,
        lr=lr,
        weight_decay=weight_decay,
        lambda_data=lambda_data,
        lambda_pde=lambda_pde,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
    )
    return val_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna によるハイパーパラメータ探索")
    parser.add_argument("--data_dir", default=gnn.DATA_DIR, help="学習データのディレクトリ")
    parser.add_argument("--trials", type=int, default=10, help="試行回数")
    parser.add_argument("--num_epochs", type=int, default=200, help="1 試行あたりのエポック数")
    parser.add_argument(
        "--max_num_cases",
        type=int,
        default=30,
        help="探索時に使用する (time, rank) ペアの最大件数",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="探索時の train/val 分割比率",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="乱数シード（Optuna と学習の再現性用）",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=Path("optuna_trials_history.tsv"),
        help="試行番号と検証誤差を逐次追記するログファイルのパス",
    )
    parser.add_argument(
        "--lazy_loading",
        action="store_true",
        default=True,
        help="遅延GPU転送を有効化（デフォルト: 有効）",
    )
    parser.add_argument(
        "--no_lazy_loading",
        action="store_true",
        help="遅延GPU転送を無効化",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="混合精度学習 (AMP) を有効化（デフォルト: 有効）",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="混合精度学習 (AMP) を無効化",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help="データキャッシュを有効化（デフォルト: 有効）",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="データキャッシュを無効化（毎回ファイルから読み込む）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache",
        help="キャッシュファイルの保存先ディレクトリ（デフォルト: .cache）",
    )
    parser.add_argument(
        "--lambda_gauge",
        type=float,
        default=0.01,
        help="ゲージ正則化係数（教師なし学習時の定数モード抑制用、デフォルト: 0.01）",
    )
    parser.add_argument(
        "--search_lambda_gauge",
        action="store_true",
        help="ゲージ正則化係数も Optuna で探索する",
    )

    args = parser.parse_args()

    # メモリ効率化オプションの設定
    gnn.USE_LAZY_LOADING = not args.no_lazy_loading
    gnn.USE_AMP = not args.no_amp

    # データキャッシュオプションの設定
    gnn.USE_DATA_CACHE = not args.no_cache
    gnn.CACHE_DIR = args.cache_dir

    sampler = optuna.samplers.TPESampler(seed=args.random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # ログファイルを探索開始前に準備
    _initialize_log_file(args.log_file)

    # Optuna 探索本体
    study.optimize(
        lambda trial: objective(
            trial,
            args.data_dir,
            args.num_epochs,
            args.max_num_cases,
            args.train_fraction,
            args.random_seed,
            args.log_file,
            args.lambda_gauge,
            args.search_lambda_gauge,
        ),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    # ベストトライアル情報を表示
    print("=== Best trial ===")
    print(f"  value (min val rel err): {study.best_trial.value:.4e}")
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # ============================================================
    # ★ ベストトライアルのハイパーパラメータで 1 回だけ再学習し、
    #    このときだけ誤差場プロットを出す
    # ============================================================
    best = study.best_trial
    print("\n[INFO] ベストトライアルのハイパーパラメータで再学習を実行します。")

    # gnn 側のグローバル設定を引数に合わせて再セット
    gnn.NUM_EPOCHS      = args.num_epochs
    gnn.MAX_NUM_CASES   = args.max_num_cases
    gnn.TRAIN_FRACTION  = args.train_fraction
    gnn.RANDOM_SEED     = args.random_seed

    # ベストトライアルのハイパーパラメータを gnn 側に反映
    gnn.LR              = best.params["lr"]
    gnn.WEIGHT_DECAY    = best.params["weight_decay"]
    gnn.LAMBDA_DATA     = best.params["lambda_data"]
    gnn.LAMBDA_PDE      = best.params["lambda_pde"]
    gnn.HIDDEN_CHANNELS = best.params["hidden_channels"]
    gnn.NUM_LAYERS      = best.params["num_layers"]

    # ★ この呼び出しだけ誤差場プロットを有効化
    #   enable_plot はお好みで True/False を選んでください
    gnn.train_gnn_auto_trainval_pde_weighted(
        args.data_dir,
        enable_plot=False,          # 学習曲線のポップアップが不要なら False
        return_history=False,       # ここでは履歴は使わないので False
        enable_error_plots=True,    # ← ベスト設定のときだけ誤差場 PNG を出力
    )

    print("[INFO] ベスト設定での再学習と誤差場プロット出力が完了しました。")


if __name__ == "__main__":
    main()
