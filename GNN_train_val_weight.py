#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_gnn_auto_trainval_pde_weighted.py

- DATA_DIR 内から自動的に pEqn_*_rank*.dat を走査し、
  全ての (time, rank) ペアを最大 MAX_NUM_CASES 件まで自動生成。
- 複数プロセス（rank）のデータを統合して学習。
- その (time, rank) リストを train/val に分割して学習。
- 損失は data loss (相対二乗誤差) + mesh-quality-weighted PDE loss。
- 学習中に、loss / data_loss / PDE_loss / rel_err_train / rel_err_val を
  リアルタイムにポップアップ表示。

"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用（projection="3d"で内部的に使用）
import time
from datetime import datetime
import pickle
import hashlib
# 日本語フォントを指定（インストール済みのものから選ぶ）
plt.rcParams['font.family'] = 'IPAexGothic'    # or 'Noto Sans CJK JP' など
# マイナス記号が文字化けする場合の対策
plt.rcParams['axes.unicode_minus'] = False

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    raise RuntimeError(
        "torch_geometric がインストールされていません。"
        "pip install torch-geometric などでインストールしてください。"
    )

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------

DATA_DIR       = "./data"
OUTPUT_DIR     = "./"
NUM_EPOCHS     = 1000
LR             = 1e-3
WEIGHT_DECAY   = 1e-5
MAX_NUM_CASES  = 100   # 自動検出した time のうち先頭 MAX_NUM_CASES 件を使用
TRAIN_FRACTION = 0.8   # 全ケースのうち train に使う割合
HIDDEN_CHANNELS = 64
NUM_LAYERS      = 4

# 学習率スケジューラ（ReduceLROnPlateau）用パラメータ
USE_LR_SCHEDULER = True
LR_SCHED_FACTOR = 0.5
LR_SCHED_PATIENCE = 20
LR_SCHED_MIN_LR = 1e-6

# メモリ効率化オプション
USE_LAZY_LOADING = True   # データをCPUに保持し、使用時のみGPUへ転送
USE_AMP = True            # 混合精度学習（Automatic Mixed Precision）を有効化

# データキャッシュオプション（Optuna等での繰り返し学習を高速化）
USE_DATA_CACHE = True     # データをキャッシュファイルに保存し、2回目以降は高速ロード
CACHE_DIR = ".cache"      # キャッシュファイルの保存先ディレクトリ

LAMBDA_DATA = 0.1
LAMBDA_PDE  = 0.0001
LAMBDA_LAPLACIAN = 0.0001  # オートディファレンスで計算したラプラシアン損失の重み
LAMBDA_BC = 0.01  # 境界条件損失（WALL_FACES 利用）の重み
LAMBDA_GAUGE = 0.01  # ゲージ正則化係数（教師なし学習時の定数モード抑制用）

W_PDE_MAX = 10.0  # w_pde の最大値

# オートディファレンスでのラプラシアン損失を有効化するかどうか
USE_AUTODIFF_LAPLACIAN_LOSS = False
# WALL_FACES による境界条件損失を有効化するかどうか
USE_BC_LOSS = True

EPS_DATA = 1e-12  # データ損失用 eps
EPS_RES  = 1e-12  # 残差正規化用 eps
EPS_PLOT = 1e-12  # ★ログプロット用の下限値

RANDOM_SEED = 42  # train/val をランダム分割するためのシード

# 可視化の更新間隔（エポック）
PLOT_INTERVAL = 10

# 誤差場可視化用の設定
MAX_ERROR_PLOT_CASES_TRAIN = 3   # train ケースで誤差図を出す最大件数
MAX_ERROR_PLOT_CASES_VAL   = 3   # val ケースで誤差図を出す最大件数
MAX_POINTS_3D_SCATTER      = 50000  # 3D散布図でプロットする最大セル数（それ以上ならランダムサンプリング）
YSLICE_FRACTIONAL_HALF_WIDTH = 0.05  # y中央断面として扱う帯の半幅（全高さに対する 5%）

# ログファイル用
LOGGER_FILE = None

def log_print(msg: str):
    """標準出力とログファイル（あれば）の両方に同じメッセージを出力する。"""
    print(msg)
    global LOGGER_FILE
    if LOGGER_FILE is not None:
        print(msg, file=LOGGER_FILE)
        LOGGER_FILE.flush()

# ------------------------------------------------------------
# ユーティリティ: (time, rank) ペアリスト自動検出
# ------------------------------------------------------------

import re
import glob

def find_time_rank_list(data_dir: str):
    """
    DATA_DIR/processor*/gnn/ 内から全ての pEqn_{time}_rank{rank}.dat を走査し、
    対応する A_csr_{time}.dat が存在する (time, rank, gnn_dir) タプルのリストを返す。
    x_{time}_rank{rank}.dat は教師なし学習モードでは省略可能。

    ディレクトリ構造:
        data/
        ├── processor2/gnn/
        │   ├── A_csr_{time}.dat
        │   ├── pEqn_{time}_rank2.dat
        │   └── x_{time}_rank2.dat  # 省略可
        ├── processor4/gnn/
        │   └── ...
        └── ...

    Returns:
        tuple: (time_rank_tuples, missing_files_info)
            - time_rank_tuples: 有効な (time, rank, gnn_dir) タプルのリスト
            - missing_files_info: 見つからなかったファイルの情報（辞書）
    """
    time_rank_tuples = []
    pattern = re.compile(r"^pEqn_(.+)_rank(\d+)\.dat$")

    # 見つからなかったファイルを追跡
    missing_pEqn = []
    missing_csr = []
    missing_x = []  # 警告用（教師なし学習では必須ではない）

    # data/processor*/gnn/ を探索
    gnn_dirs = glob.glob(os.path.join(data_dir, "processor*", "gnn"))

    if not gnn_dirs:
        # gnn ディレクトリ自体が見つからない場合
        return [], {"no_gnn_dirs": True}

    for gnn_dir in gnn_dirs:
        if not os.path.isdir(gnn_dir):
            continue

        for fn in os.listdir(gnn_dir):
            match = pattern.match(fn)
            if not match:
                continue

            time_str = match.group(1)
            rank_str = match.group(2)

            x_path   = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")
            # CSR ファイルは A_csr_{time}.dat または A_csr_{time}_rank{rank}.dat の両形式に対応
            csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
            csr_path_with_rank = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

            has_csr = os.path.exists(csr_path) or os.path.exists(csr_path_with_rank)
            has_x = os.path.exists(x_path)

            if has_csr:
                # pEqn と A_csr があれば有効（x は教師なし学習では省略可）
                time_rank_tuples.append((time_str, rank_str, gnn_dir))
                if not has_x:
                    missing_x.append(x_path)
            else:
                missing_csr.append(csr_path)

    # time の数値順、次に rank の数値順でソート
    time_rank_tuples = sorted(
        set(time_rank_tuples),
        key=lambda tr: (float(tr[0]), int(tr[1]))
    )

    missing_info = {
        "missing_pEqn": missing_pEqn,
        "missing_csr": missing_csr,
        "missing_x": missing_x,
    }

    return time_rank_tuples, missing_info


# ------------------------------------------------------------
# データキャッシュ機能
# ------------------------------------------------------------

def _compute_cache_key(data_dir: str, time_rank_tuples: list) -> str:
    """
    キャッシュのキー（ハッシュ）を計算する。
    データディレクトリと (time, rank, gnn_dir) タプルのリストから一意のキーを生成。
    """
    key_str = data_dir + "|" + str(sorted(time_rank_tuples))
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _get_cache_path(data_dir: str, time_rank_tuples: list) -> str:
    """キャッシュファイルのパスを取得する。"""
    cache_key = _compute_cache_key(data_dir, time_rank_tuples)
    return os.path.join(CACHE_DIR, f"raw_cases_{cache_key}.pkl")


def _is_cache_valid(cache_path: str, time_rank_tuples: list) -> bool:
    """
    キャッシュが有効かどうかを確認する。
    - キャッシュファイルが存在するか
    - ソースファイルよりキャッシュが新しいか
    """
    if not os.path.exists(cache_path):
        return False

    cache_mtime = os.path.getmtime(cache_path)

    # 各ソースファイルの最終更新時刻をチェック
    for time_str, rank_str, gnn_dir in time_rank_tuples:
        p_path = os.path.join(gnn_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
        x_path = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")
        csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
        csr_path_with_rank = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

        for path in [p_path, x_path]:
            if os.path.exists(path) and os.path.getmtime(path) > cache_mtime:
                return False

        # CSR ファイルは両形式に対応
        if os.path.exists(csr_path) and os.path.getmtime(csr_path) > cache_mtime:
            return False
        if os.path.exists(csr_path_with_rank) and os.path.getmtime(csr_path_with_rank) > cache_mtime:
            return False

    return True


def save_raw_cases_to_cache(raw_cases: list, cache_path: str) -> None:
    """raw_cases をキャッシュファイルに保存する。"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(raw_cases, f, protocol=pickle.HIGHEST_PROTOCOL)
    log_print(f"[CACHE] データを {cache_path} にキャッシュしました")


def load_raw_cases_from_cache(cache_path: str) -> list:
    """キャッシュファイルから raw_cases を読み込む。"""
    with open(cache_path, "rb") as f:
        raw_cases = pickle.load(f)
    log_print(f"[CACHE] キャッシュ {cache_path} からデータを読み込みました")
    return raw_cases


def compute_affine_fit(x_true_tensor, x_pred_tensor):
    """
    x_true ≈ a * x_pred + b となるように、
    最小二乗で a, b を求める簡易診断用関数。

    Parameters
    ----------
    x_true_tensor : torch.Tensor, shape (N,)
        物理スケールの真値（正規化解除後）
    x_pred_tensor : torch.Tensor, shape (N,)
        物理スケールの予測値（正規化解除後）

    Returns
    -------
    a : float
        最適スケール係数
    b : float
        最適バイアス
    rmse_before : float
        補正前 RMSE = sqrt(mean((x_pred - x_true)^2))
    rmse_after : float
        補正後 RMSE = sqrt(mean((a*x_pred + b - x_true)^2))
    """
    # CPU / numpy に変換して 1 次元にフラット化
    xp = x_pred_tensor.detach().cpu().double().view(-1).numpy()
    yt = x_true_tensor.detach().cpu().double().view(-1).numpy()

    n = xp.size
    if n == 0:
        return 1.0, 0.0, float("nan"), float("nan")

    sx = xp.sum()
    sy = yt.sum()
    sxx = (xp * xp).sum()
    sxy = (xp * yt).sum()

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        # x_pred がほぼ定数の場合はスケールをいじれないので、そのままとみなす
        a = 1.0
        b = 0.0
    else:
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n

    rmse_before = float(np.sqrt(((xp - yt) ** 2).mean()))
    rmse_after = float(np.sqrt(((a * xp + b - yt) ** 2).mean()))

    return a, b, rmse_before, rmse_after


# ------------------------------------------------------------
# pEqn + CSR + x_true 読み込み
# ------------------------------------------------------------

def load_case_with_csr(gnn_dir: str, time_str: str, rank_str: str):
    """
    指定された gnn_dir から (time, rank) に対応するデータを読み込む。

    ファイル形式:
        - pEqn_{time}_rank{rank}.dat
        - x_{time}_rank{rank}.dat
        - A_csr_{time}.dat または A_csr_{time}_rank{rank}.dat
    """
    p_path   = os.path.join(gnn_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
    x_path   = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")

    # CSR ファイルは両形式に対応
    csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
    if not os.path.exists(csr_path):
        csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

    if not os.path.exists(p_path):
        raise FileNotFoundError(p_path)
    # x ファイルは存在しなくてもよい（教師なし学習モード）
    has_x_true = os.path.exists(x_path)
    if not os.path.exists(csr_path):
        raise FileNotFoundError(csr_path)

    with open(p_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    try:
        header_nc = lines[0].split()
        header_nf = lines[1].split()
        assert header_nc[0] == "nCells"
        assert header_nf[0] == "nFaces"
        nCells = int(header_nc[1])
    except Exception as e:
        raise RuntimeError(f"nCells/nFaces ヘッダの解釈に失敗しました: {p_path}\n{e}")

    try:
        idx_cells = next(i for i, ln in enumerate(lines) if ln.startswith("CELLS"))
        idx_edges = next(i for i, ln in enumerate(lines) if ln.startswith("EDGES"))
    except StopIteration:
        raise RuntimeError(f"CELLS/EDGES セクションが見つかりません: {p_path}")

    idx_wall = None
    for i, ln in enumerate(lines):
        if ln.startswith("WALL_FACES"):
            idx_wall = i
            break
    if idx_wall is None:
        idx_wall = len(lines)

    cell_lines = lines[idx_cells + 1: idx_edges]
    edge_lines = lines[idx_edges + 1: idx_wall]
    wall_lines = lines[idx_wall:]

    if len(cell_lines) != nCells:
        log_print(f"[WARN] nCells={nCells} と CELLS 行数={len(cell_lines)} が異なります (time={time_str}).")

    feats_np = np.zeros((len(cell_lines), 13), dtype=np.float32)
    b_np     = np.zeros(len(cell_lines), dtype=np.float32)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 14:
            raise RuntimeError(f"CELLS 行の列数が足りません: {ln}")
        cell_id = int(parts[0])
        xcoord  = float(parts[1])
        ycoord  = float(parts[2])
        zcoord  = float(parts[3])
        diag    = float(parts[4])
        b_val   = float(parts[5])
        skew    = float(parts[6])
        non_ortho  = float(parts[7])
        aspect     = float(parts[8])
        diag_con   = float(parts[9])
        V          = float(parts[10])
        h          = float(parts[11])
        size_jump  = float(parts[12])
        Co         = float(parts[13])

        if not (0 <= cell_id < len(cell_lines)):
            raise RuntimeError(f"cell_id の範囲がおかしいです: {cell_id}")

        feats_np[cell_id, :] = np.array(
            [
                xcoord, ycoord, zcoord,
                diag, b_val, skew, non_ortho, aspect,
                diag_con, V, h, size_jump, Co
            ],
            dtype=np.float32
        )
        b_np[cell_id] = b_val

    e_src = []
    e_dst = []
    for ln in edge_lines:
        parts = ln.split()
        if parts[0] == "WALL_FACES":
            break
        if len(parts) != 5:
            raise RuntimeError(f"EDGES 行の列数が 5 ではありません: {ln}")
        lower = int(parts[1])
        upper = int(parts[2])
        if not (0 <= lower < len(cell_lines) and 0 <= upper < len(cell_lines)):
            raise RuntimeError(f"lower/upper の cell index が範囲外です: {ln}")

        e_src.append(lower)
        e_dst.append(upper)
        e_src.append(upper)
        e_dst.append(lower)

    edge_index_np = np.vstack([
        np.array(e_src, dtype=np.int64),
        np.array(e_dst, dtype=np.int64)
    ])

    # WALL_FACES セクション（境界条件）をパース
    wall_bc_cells = []
    wall_bc_values = []
    wall_bc_weights = []
    for ln in wall_lines:
        parts = ln.split()
        if not parts or parts[0] != "WALL_FACES":
            continue

        if len(parts) < 2:
            log_print(f"[WARN] WALL_FACES 行の列数が不足しています: {ln}")
            continue

        try:
            cell_id = int(parts[1])
        except ValueError:
            log_print(f"[WARN] WALL_FACES の cell_id を解釈できません: {ln}")
            continue

        if not (0 <= cell_id < len(cell_lines)):
            log_print(f"[WARN] WALL_FACES の cell_id が範囲外です: {cell_id}")
            continue

        bc_val = float(parts[2]) if len(parts) >= 3 else 0.0
        bc_weight = float(parts[3]) if len(parts) >= 4 else 1.0

        wall_bc_cells.append(cell_id)
        wall_bc_values.append(bc_val)
        wall_bc_weights.append(bc_weight)

    wall_bc_index_np = np.array(wall_bc_cells, dtype=np.int64)
    wall_bc_value_np = np.array(wall_bc_values, dtype=np.float32)
    wall_bc_weight_np = np.array(wall_bc_weights, dtype=np.float32)

    # x ファイルが存在する場合のみ読み込み（教師あり学習）
    # 存在しない場合は None（教師なし学習 / PINNs モード）
    if has_x_true:
        x_true_np = np.zeros(len(cell_lines), dtype=np.float32)
        with open(x_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) != 2:
                    raise RuntimeError(f"x_*.dat の行形式が想定外です: {ln}")
                cid = int(parts[0])
                val = float(parts[1])
                if not (0 <= cid < len(cell_lines)):
                    raise RuntimeError(f"x_*.dat の cell id が範囲外です: {cid}")
                x_true_np[cid] = val
    else:
        x_true_np = None

    with open(csr_path, "r") as f:
        csr_lines = [ln.strip() for ln in f if ln.strip()]

    try:
        h0 = csr_lines[0].split()
        h1 = csr_lines[1].split()
        h2 = csr_lines[2].split()
        assert h0[0] == "nRows"
        assert h1[0] == "nCols"
        assert h2[0] == "nnz"
        nRows = int(h0[1])
        nCols = int(h1[1])
        nnz   = int(h2[1])
    except Exception as e:
        raise RuntimeError(f"A_csr_{time_str}.dat のヘッダ解釈に失敗しました: {csr_path}\n{e}")

    if nRows != nCells:
        log_print(f"[WARN] CSR nRows={nRows} と pEqn nCells={nCells} が異なります (time={time_str}).")

    try:
        idx_rowptr = next(i for i, ln in enumerate(csr_lines) if ln.startswith("ROW_PTR"))
        idx_colind = next(i for i, ln in enumerate(csr_lines) if ln.startswith("COL_IND"))
        idx_vals   = next(i for i, ln in enumerate(csr_lines) if ln.startswith("VALUES"))
    except StopIteration:
        raise RuntimeError(f"ROW_PTR/COL_IND/VALUES が見つかりません: {csr_path}")

    row_ptr_str = csr_lines[idx_rowptr + 1].split()
    col_ind_str = csr_lines[idx_colind + 1].split()
    vals_str    = csr_lines[idx_vals + 1].split()

    if len(row_ptr_str) != nRows + 1:
        raise RuntimeError(
            f"ROW_PTR の長さが nRows+1 と一致しません: len={len(row_ptr_str)}, nRows={nRows}"
        )
    if len(col_ind_str) != nnz:
        raise RuntimeError(
            f"COL_IND の長さが nnz と一致しません: len={len(col_ind_str)}, nnz={nnz}"
        )
    if len(vals_str) != nnz:
        raise RuntimeError(
            f"VALUES の長さが nnz と一致しません: len={len(vals_str)}, nnz={nnz}"
        )

    row_ptr_np = np.array(row_ptr_str, dtype=np.int64)
    col_ind_np = np.array(col_ind_str, dtype=np.int64)
    vals_np    = np.array(vals_str,    dtype=np.float32)

    row_idx_np = np.empty(nnz, dtype=np.int64)
    for i in range(nRows):
        start = row_ptr_np[i]
        end   = row_ptr_np[i+1]
        row_idx_np[start:end] = i

    return {
        "time": time_str,
        "rank": rank_str,
        "gnn_dir": gnn_dir,
        "feats_np": feats_np,
        "edge_index_np": edge_index_np,
        "x_true_np": x_true_np,
        "has_x_true": has_x_true,  # 教師データの有無フラグ
        "b_np": b_np,
        "row_ptr_np": row_ptr_np,
        "col_ind_np": col_ind_np,
        "vals_np": vals_np,
        "row_idx_np": row_idx_np,
        "wall_bc_index_np": wall_bc_index_np,
        "wall_bc_value_np": wall_bc_value_np,
        "wall_bc_weight_np": wall_bc_weight_np,
    }

# ------------------------------------------------------------
# GNN
# ------------------------------------------------------------

class SimpleSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, 1))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x.view(-1)

# ------------------------------------------------------------
# CSR Ax
# ------------------------------------------------------------

def matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x):
    """
    CSR 形式の疎行列とベクトルの積を計算する。
    AMP 使用時に型の不一致が発生する場合は、自動的に型を揃える。
    """
    # AMP 使用時、x が half (FP16) で vals が float (FP32) の場合がある
    # 計算精度を保つため、x を vals の型に揃える
    if x.dtype != vals.dtype:
        x = x.to(vals.dtype)

    y = torch.zeros_like(x)
    y.index_add_(0, row_idx, vals * x[col_ind])
    return y

# ------------------------------------------------------------
# オートディファレンスによるラプラシアン損失
# ------------------------------------------------------------

def compute_autodiff_laplacian_loss(x_pred: torch.Tensor,
                                    coords: torch.Tensor,
                                    b: torch.Tensor,
                                    weight: torch.Tensor) -> torch.Tensor:
    """
    予測値 x_pred に対して、座標に対するオートディファレンスでラプラシアンを計算し、
    RHS (b) との差を相対残差²として返す。

    Parameters
    ----------
    x_pred : torch.Tensor
        予測スカラー場（形状: [N]）
    coords : torch.Tensor
        座標特徴（形状: [N, 3]）。requires_grad=True である必要がある。
    b : torch.Tensor
        PDE の RHS ベクトル（形状: [N]）
    weight : torch.Tensor
        PDE 重み（形状: [N]）。w_pde を想定。
    """

    # 1次導関数
    grad = torch.autograd.grad(
        outputs=x_pred,
        inputs=coords,
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 2次導関数（各軸の 2階微分の和 = ラプラシアン）
    laplacian_terms = []
    for i in range(grad.shape[1]):
        grad_i = grad[:, i]
        second_deriv = torch.autograd.grad(
            outputs=grad_i,
            inputs=coords,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0][:, i]
        laplacian_terms.append(second_deriv)

    laplacian = torch.stack(laplacian_terms, dim=1).sum(dim=1)

    # ラプラシアン - b を相対残差²で評価
    residual = laplacian - b
    sqrt_w = torch.sqrt(weight)
    wr = sqrt_w * residual
    wb = sqrt_w * b
    norm_wr = torch.norm(wr)
    norm_wb = torch.norm(wb) + EPS_RES
    R_laplace = norm_wr / norm_wb
    return R_laplace * R_laplace

# ------------------------------------------------------------
# メッシュ品質 w_pde
# ------------------------------------------------------------

def build_w_pde_from_feats(feats_np: np.ndarray,
                           w_pde_max: float = W_PDE_MAX) -> np.ndarray:
    """
    メッシュ品質に基づくPDE損失の重みを計算
    """
    # メトリクス抽出
    skew      = feats_np[:, 5]
    non_ortho = feats_np[:, 6]
    aspect    = feats_np[:, 7]
    size_jump = feats_np[:, 11]

    # 基準値
    SKEW_REF      = 0.2
    NONORTH_REF   = 10.0
    ASPECT_REF    = 5.0
    SIZEJUMP_REF  = 1.5

    # 正規化（0.0〜5.0にクリップ）
    q_skew      = np.clip(skew      / (SKEW_REF + 1e-12),     0.0, 5.0)
    q_non_ortho = np.clip(non_ortho / (NONORTH_REF + 1e-12),  0.0, 5.0)
    q_aspect    = np.clip(aspect    / (ASPECT_REF + 1e-12),   0.0, 5.0)
    q_sizeJump  = np.clip(size_jump / (SIZEJUMP_REF + 1e-12), 0.0, 5.0)

    # 線形結合
    w_raw = (
        1.0
        + 1.0 * (q_skew      - 1.0)
        + 1.0 * (q_non_ortho - 1.0)
        + 1.0 * (q_aspect    - 1.0)
        + 1.0 * (q_sizeJump  - 1.0)
    )

    # クリップ
    w_clipped = np.clip(w_raw, 1.0, w_pde_max)

    return w_clipped.astype(np.float32)

# ------------------------------------------------------------
# raw_case → torch case への変換ヘルパ
# ------------------------------------------------------------

def convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device, lazy_load=False):
    """
    raw_case を torch テンソルに変換する。

    Parameters
    ----------
    lazy_load : bool
        True の場合、データを CPU 上に保持し、GPU への転送は行わない。
        学習時に move_case_to_device() で必要なときだけ GPU に転送する。
    """
    feats_np  = rc["feats_np"]
    x_true_np = rc["x_true_np"]
    has_x_true = rc.get("has_x_true", x_true_np is not None)
    wall_bc_index_np = rc.get("wall_bc_index_np", np.array([], dtype=np.int64))
    wall_bc_value_np = rc.get("wall_bc_value_np", np.array([], dtype=np.float32))
    wall_bc_weight_np = rc.get("wall_bc_weight_np", np.array([], dtype=np.float32))

    feats_norm = (feats_np - feat_mean) / feat_std

    # x_true が存在する場合のみ正規化（教師あり学習）
    if has_x_true and x_true_np is not None:
        x_true_norm_np = (x_true_np - x_mean) / x_std
    else:
        x_true_norm_np = None

    # ★ ここで w_pde_np を計算
    w_pde_np = build_w_pde_from_feats(feats_np)

    # lazy_load が True の場合は CPU に保持、False の場合は直接 device へ
    target_device = torch.device("cpu") if lazy_load else device

    feats       = torch.from_numpy(feats_norm).float().to(target_device)
    edge_index  = torch.from_numpy(rc["edge_index_np"]).long().to(target_device)

    # x_true が存在する場合のみテンソル化
    if has_x_true and x_true_np is not None:
        x_true      = torch.from_numpy(x_true_np).float().to(target_device)
        x_true_norm = torch.from_numpy(x_true_norm_np).float().to(target_device)
    else:
        x_true      = None
        x_true_norm = None

    b       = torch.from_numpy(rc["b_np"]).float().to(target_device)
    row_ptr = torch.from_numpy(rc["row_ptr_np"]).long().to(target_device)
    col_ind = torch.from_numpy(rc["col_ind_np"]).long().to(target_device)
    vals    = torch.from_numpy(rc["vals_np"]).float().to(target_device)
    row_idx = torch.from_numpy(rc["row_idx_np"]).long().to(target_device)

    w_pde = torch.from_numpy(w_pde_np).float().to(target_device)

    # 境界条件（WALL_FACES）
    if wall_bc_index_np.size > 0:
        wall_bc_index = torch.from_numpy(wall_bc_index_np).long().to(target_device)
        wall_bc_value = torch.from_numpy(wall_bc_value_np).float().to(target_device)
        wall_bc_weight = torch.from_numpy(wall_bc_weight_np).float().to(target_device)
    else:
        wall_bc_index = None
        wall_bc_value = None
        wall_bc_weight = None

    return {
        "time": rc["time"],
        "rank": rc["rank"],
        "gnn_dir": rc["gnn_dir"],
        "feats": feats,
        "edge_index": edge_index,
        "x_true": x_true,
        "x_true_norm": x_true_norm,
        "has_x_true": has_x_true,  # 教師データの有無フラグ
        "b": b,
        "row_ptr": row_ptr,
        "col_ind": col_ind,
        "vals": vals,
        "row_idx": row_idx,
        "w_pde": w_pde,
        "w_pde_np": w_pde_np,  # ★ 分布ログ用に numpy を保持しておく
        "wall_bc_index": wall_bc_index,
        "wall_bc_value": wall_bc_value,
        "wall_bc_weight": wall_bc_weight,

        # ★ 誤差場可視化用に元の座標・品質指標も持たせる
        "coords_np": feats_np[:, 0:3].copy(),   # [x, y, z]
        "skew_np": feats_np[:, 5].copy(),
        "non_ortho_np": feats_np[:, 6].copy(),
        "aspect_np": feats_np[:, 7].copy(),
        "size_jump_np": feats_np[:, 11].copy(),
    }


def move_case_to_device(cs, device):
    """
    ケースデータを指定デバイスに転送する（遅延ロード用）。
    non_blocking=True で非同期転送を行い、オーバーヘッドを軽減。
    x_true が None の場合（教師なし学習）も対応。
    """
    x_true = cs["x_true"]
    x_true_norm = cs["x_true_norm"]
    has_x_true = cs.get("has_x_true", x_true is not None)
    wall_bc_index = cs.get("wall_bc_index")
    wall_bc_value = cs.get("wall_bc_value")
    wall_bc_weight = cs.get("wall_bc_weight")

    return {
        "time": cs["time"],
        "rank": cs["rank"],
        "gnn_dir": cs["gnn_dir"],
        "feats": cs["feats"].to(device, non_blocking=True),
        "edge_index": cs["edge_index"].to(device, non_blocking=True),
        "x_true": x_true.to(device, non_blocking=True) if x_true is not None else None,
        "x_true_norm": x_true_norm.to(device, non_blocking=True) if x_true_norm is not None else None,
        "has_x_true": has_x_true,
        "b": cs["b"].to(device, non_blocking=True),
        "row_ptr": cs["row_ptr"].to(device, non_blocking=True),
        "col_ind": cs["col_ind"].to(device, non_blocking=True),
        "vals": cs["vals"].to(device, non_blocking=True),
        "row_idx": cs["row_idx"].to(device, non_blocking=True),
        "w_pde": cs["w_pde"].to(device, non_blocking=True),
        "w_pde_np": cs["w_pde_np"],
        "wall_bc_index": wall_bc_index.to(device, non_blocking=True) if wall_bc_index is not None else None,
        "wall_bc_value": wall_bc_value.to(device, non_blocking=True) if wall_bc_value is not None else None,
        "wall_bc_weight": wall_bc_weight.to(device, non_blocking=True) if wall_bc_weight is not None else None,
        "coords_np": cs["coords_np"],
        "skew_np": cs["skew_np"],
        "non_ortho_np": cs["non_ortho_np"],
        "aspect_np": cs["aspect_np"],
        "size_jump_np": cs["size_jump_np"],
    }

def save_error_field_plots(cs, x_pred, x_true, prefix, output_dir=OUTPUT_DIR):
    """
    誤差場 (x_pred - x_true) の 3D 散布図と、
    y ≒ 中央断面での 2D カラーマップ（誤差 vs w_pde）を保存する。

    さらに |誤差| と w_pde の簡単な統計（相関係数、誤差上位5%セルの平均w_pde など）をログ出力する。
    """
    # ---- Torch -> NumPy ----
    x_pred_np = x_pred.detach().cpu().numpy().reshape(-1)
    x_true_np = x_true.detach().cpu().numpy().reshape(-1)
    err       = x_pred_np - x_true_np
    abs_err   = np.abs(err)

    coords    = cs["coords_np"]      # (N, 3) : x, y, z
    w_pde_np  = cs["w_pde_np"]       # (N,)

    N = coords.shape[0]
    if err.shape[0] != N:
        log_print(f"    [WARN] 誤差場可視化: 座標数 N={N} と解ベクトル長={err.shape[0]} が一致しません ({prefix})。")
        return

    # ============================
    # 1) 3D 散布図 (x, y, z, color = x_pred - x_true)
    # ============================
    if N > MAX_POINTS_3D_SCATTER:
        idx = np.random.choice(N, size=MAX_POINTS_3D_SCATTER, replace=False)
        log_print(f"    [PLOT] 3D 散布図用に {N} セル中 {idx.size} セルをサンプリングしました ({prefix}).")
    else:
        idx = np.arange(N)

    xs = coords[idx, 0]
    ys = coords[idx, 1]
    zs = coords[idx, 2]
    err_sample = err[idx]

    vmax = np.max(np.abs(err_sample)) + 1e-20
    vmin = -vmax

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=err_sample, s=2, cmap="coolwarm", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("x_pred - x_true")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"誤差場 3D散布図 ({prefix})")
    fig.tight_layout()

    out3d = os.path.join(output_dir, f"error3d_{prefix}.png")
    fig.savefig(out3d, dpi=200)
    plt.close(fig)

    log_print(f"    [PLOT] 誤差場 3D 散布図を {out3d} に保存しました。")

    # ============================
    # 2) y ≒ 中央断面での 2D カラーマップ
    #    左: |x_pred - x_true|, 右: w_pde
    # ============================
    y = coords[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    if y_max > y_min:
        y_mid = 0.5 * (y_min + y_max)
        band  = YSLICE_FRACTIONAL_HALF_WIDTH * (y_max - y_min)
    else:
        # 全セル同一 y の場合
        y_mid = y_min
        band  = 1e-6

    mask = np.abs(y - y_mid) <= band
    n_slice = int(np.count_nonzero(mask))

    if n_slice < 10:
        log_print(f"    [PLOT] y≈中央断面のセル数が {n_slice} と少ないため 2D カラーマップをスキップします ({prefix}).")
    else:
        xs2       = coords[mask, 0]
        zs2       = coords[mask, 2]
        abs_err2  = abs_err[mask]
        w_pde2    = w_pde_np[mask]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sc0 = axes[0].scatter(xs2, zs2, c=abs_err2, s=5)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("z")
        axes[0].set_title("誤差場 |x_pred - x_true| (y ≒ 中央断面)")
        cbar0 = fig.colorbar(sc0, ax=axes[0])
        cbar0.set_label("|x_pred - x_true|")

        sc1 = axes[1].scatter(xs2, zs2, c=w_pde2, s=5)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        axes[1].set_title("w_pde (メッシュ品質重み, y ≒ 中央断面)")
        cbar1 = fig.colorbar(sc1, ax=axes[1])
        cbar1.set_label("w_pde")

        fig.tight_layout()
        out2d = os.path.join(output_dir, f"error2d_yMid_{prefix}.png")
        fig.savefig(out2d, dpi=200)
        plt.close(fig)

        log_print(f"    [PLOT] 誤差場と w_pde の 2D カラーマップを {out2d} に保存しました。")

    # ============================
    # 3) |誤差| と w_pde の簡単な統計
    # ============================
    if N >= 10:
        if np.std(abs_err) > 0.0 and np.std(w_pde_np) > 0.0:
            corr = float(np.corrcoef(abs_err, w_pde_np)[0, 1])
        else:
            corr = float("nan")

        top_frac = 0.05  # 誤差上位5%を見る
        k = max(1, int(top_frac * N))
        idx_sorted = np.argsort(-abs_err)  # 大きい順
        top_idx = idx_sorted[:k]

        mean_w_all = float(w_pde_np.mean())
        mean_w_top = float(w_pde_np[top_idx].mean())

        log_print(
            "    [STATS] |誤差| と w_pde の簡易統計: "
            f"corr(|err|, w_pde)={corr:.3f}, "
            f"top{int(top_frac*100)}%誤差セルの平均w_pde={mean_w_top:.3e}, "
            f"全セル平均w_pde={mean_w_all:.3e}"
        )

# ------------------------------------------------------------
# 可視化ユーティリティ
# ------------------------------------------------------------

EPS_PLOT = 1e-12  # まだ無ければ定数として追加

EPS_PLOT = 1e-12  # ログプロット用の下限値

def init_plot():
    plt.ion()
    # 横に 2 つのサブプロット（左：損失, 右：相対誤差）
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # タイトルに係数を表示
    fig.suptitle(
        f"データ損失係数: {LAMBDA_DATA:g}, PDE損失係数: {LAMBDA_PDE:g}",
        fontsize=12
    )

    # レイアウトは update_plot 側で tight_layout をかける
    return fig, axes

def update_plot(fig, axes, history):
    ax_loss, ax_rel = axes  # 左：損失, 右：相対誤差

    ax_loss.clear()
    ax_rel.clear()

    epochs = np.array(history["epoch"], dtype=np.int32)
    if len(epochs) == 0:
        return

    loss      = np.array(history["loss"], dtype=np.float64)
    data_loss = np.array(history["data_loss"], dtype=np.float64)
    pde_loss  = np.array(history["pde_loss"], dtype=np.float64)
    rel_tr    = np.array(history["rel_err_train"], dtype=np.float64)

    rel_val = np.array(
        [np.nan if v is None else float(v) for v in history["rel_err_val"]],
        dtype=np.float64
    )

    # 下限を切ってログスケールに耐えられるようにする
    loss_safe      = np.clip(loss,      EPS_PLOT, None)
    data_loss_safe = np.clip(data_loss, EPS_PLOT, None)
    pde_loss_safe  = np.clip(pde_loss,  EPS_PLOT, None)
    rel_tr_safe    = np.clip(rel_tr,    EPS_PLOT, None)

    rel_val_safe = rel_val.copy()
    mask = np.isfinite(rel_val_safe)
    rel_val_safe[mask] = np.clip(rel_val_safe[mask], EPS_PLOT, None)

    # --- 左グラフ：損失系（総損失・データ損失・PDE損失） ---
    ax_loss.plot(epochs, loss_safe,      label="総損失",      linewidth=2)
    ax_loss.plot(epochs, data_loss_safe, label="データ損失",  linewidth=1.5, linestyle="--")
    ax_loss.plot(epochs, pde_loss_safe,  label="PDE損失",    linewidth=1.5, linestyle="--")

    ax_loss.set_xlabel("エポック数")
    ax_loss.set_ylabel("損失")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # --- 右グラフ：相対誤差（train/val） ---
    ax_rel.plot(epochs, rel_tr_safe,  label="相対誤差（訓練データ）", linewidth=1.5)
    ax_rel.plot(epochs, rel_val_safe, label="相対誤差（テストデータ）", linewidth=1.5)

    ax_rel.set_xlabel("エポック数")
    ax_rel.set_ylabel("相対誤差")
    ax_rel.set_yscale("log")
    ax_rel.grid(True, alpha=0.3)
    ax_rel.legend()

    # 図全体のレイアウト調整
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])

    plt.pause(0.01)

# ------------------------------------------------------------
# メイン: train/val 分離版
# ------------------------------------------------------------

def train_gnn_auto_trainval_pde_weighted(
    data_dir: str,
    *,
    enable_plot: bool = True,
    return_history: bool = False,
    enable_error_plots: bool = False,  # ★ 追加：誤差場プロットを出すかどうか
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global LOGGER_FILE

    # --- ログファイルと実行時間計測のセットアップ ---
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 係数をファイル名用のタグに変換（例: 0.1 → "0p1", 1e-4 → "0p0001" など）
    lambda_data_tag = str(LAMBDA_DATA).replace('.', 'p')
    lambda_pde_tag  = str(LAMBDA_PDE).replace('.', 'p')

    log_filename = (
#        f"gnn_train_log_"
        f"log_"
        f"DATA{lambda_data_tag}_"
#        f"LP{lambda_pde_tag}_"
        f"PDE{lambda_pde_tag}.txt"
#        f"{timestamp}.txt"
    )
    log_path = os.path.join(OUTPUT_DIR, log_filename)

    LOGGER_FILE = open(log_path, "w", buffering=1)  # 行バッファ

    start_time = time.time()

    log_print(f"[INFO] Logging to {log_path}")
    log_print(f"[INFO] device = {device}")

    # --- (time, rank, gnn_dir) タプルリスト検出 & 分割 ---
    all_time_rank_tuples, missing_info = find_time_rank_list(data_dir)

    if not all_time_rank_tuples:
        # 見つからなかったファイルに応じてエラーメッセージを生成
        if missing_info.get("no_gnn_dirs"):
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ ディレクトリが見つかりませんでした。"
            )

        error_messages = []
        if missing_info.get("missing_csr"):
            # CSR ファイルが見つからなかった場合
            error_messages.append("A_csr_*.dat が見つかりませんでした。")
        if not missing_info.get("missing_csr"):
            # pEqn ファイルが見つからなかった場合（CSR はあるのに pEqn がない）
            error_messages.append("pEqn_*_rank*.dat が見つかりませんでした。")

        if error_messages:
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ 内に " + " ".join(error_messages)
            )
        else:
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ 内に有効なデータが見つかりませんでした。"
            )

    # x ファイルが見つからなかった場合は警告を表示（教師なし学習モードで続行）
    if missing_info.get("missing_x"):
        num_missing_x = len(missing_info["missing_x"])
        log_print(f"[WARN] x_*_rank*.dat が {num_missing_x} 件見つかりませんでした。教師なし学習モードで続行します。")

    # 検出されたランクの一覧をログ出力
    all_ranks = sorted(set(r for _, r, _ in all_time_rank_tuples), key=int)
    all_times_unique = sorted(set(t for t, _, _ in all_time_rank_tuples), key=float)
    all_gnn_dirs = sorted(set(g for _, _, g in all_time_rank_tuples))
    log_print(f"[INFO] 検出された rank 一覧: {all_ranks}")
    log_print(f"[INFO] 検出された time 一覧: {all_times_unique[:10]}{'...' if len(all_times_unique) > 10 else ''}")
    log_print(f"[INFO] 検出された gnn_dir 数: {len(all_gnn_dirs)}")

    # 以降の print(...) はすべて log_print(...) に置き換え
    random.seed(RANDOM_SEED)
    random.shuffle(all_time_rank_tuples)

    all_time_rank_tuples = all_time_rank_tuples[:MAX_NUM_CASES]
    n_total = len(all_time_rank_tuples)
    n_train = max(1, int(n_total * TRAIN_FRACTION))
    n_val   = n_total - n_train

    tuples_train = all_time_rank_tuples[:n_train]
    tuples_val   = all_time_rank_tuples[n_train:]

    log_print(f"[INFO] 検出された (time, rank) ペア数 (使用分) = {n_total}")
    log_print(f"[INFO] train: {n_train} cases, val: {n_val} cases (TRAIN_FRACTION={TRAIN_FRACTION})")
    log_print("=== 使用する train ケース (time, rank) ===")
    for t, r, g in tuples_train:
        log_print(f"  time={t}, rank={r}")
    log_print("=== 使用する val ケース (time, rank) ===")
    if tuples_val:
        for t, r, g in tuples_val:
            log_print(f"  time={t}, rank={r}")
    else:
        log_print("  (val ケースなし)")
    log_print("===========================================")


    # --- raw ケース読み込み（train + val 両方） ---
    # キャッシュが有効な場合はキャッシュから読み込み、そうでなければファイルから読み込んでキャッシュ
    raw_cases_all = []
    cache_path = _get_cache_path(data_dir, all_time_rank_tuples) if USE_DATA_CACHE else None

    if USE_DATA_CACHE and _is_cache_valid(cache_path, all_time_rank_tuples):
        # キャッシュから読み込み（高速）
        raw_cases_all = load_raw_cases_from_cache(cache_path)
    else:
        # ファイルから読み込み
        for t, r, g in all_time_rank_tuples:
            log_print(f"[LOAD] time={t}, rank={r} のグラフ+PDE情報を読み込み中...")
            rc = load_case_with_csr(g, t, r)
            raw_cases_all.append(rc)

        # キャッシュに保存
        if USE_DATA_CACHE:
            save_raw_cases_to_cache(raw_cases_all, cache_path)

    # train/val に分割
    raw_cases_train = []
    raw_cases_val   = []
    train_set = set(tuples_train)

    for rc in raw_cases_all:
        key = (rc["time"], rc["rank"], rc["gnn_dir"])
        if key in train_set:
            raw_cases_train.append(rc)
        else:
            raw_cases_val.append(rc)

    # 特徴量次元数の一貫性チェック（セル数は rank ごとに異なる可能性あり）
    nFeat = raw_cases_train[0]["feats_np"].shape[1]
    for rc in raw_cases_train + raw_cases_val:
        if rc["feats_np"].shape[1] != nFeat:
            raise RuntimeError("全ケースで nFeatures が一致していません。")

    total_cells = sum(rc["feats_np"].shape[0] for rc in raw_cases_train + raw_cases_val)
    log_print(f"[INFO] nFeatures = {nFeat}, 総セル数 (全ケース合計) = {total_cells}")

    # --- 教師なし学習モード判定 ---
    # 全ケースの has_x_true を確認
    cases_with_x = [rc for rc in (raw_cases_train + raw_cases_val) if rc.get("has_x_true", False)]
    unsupervised_mode = len(cases_with_x) == 0

    if unsupervised_mode:
        log_print("[INFO] *** 教師なし学習モード（PINNs）: x_*_rank*.dat が見つかりません ***")
        log_print("[INFO] *** 損失関数は PDE 損失のみを使用します ***")

    # --- グローバル正規化: train+val 全体で統計を取る ---
    all_feats = np.concatenate(
        [rc["feats_np"] for rc in (raw_cases_train + raw_cases_val)], axis=0
    )

    feat_mean = all_feats.mean(axis=0, keepdims=True)
    feat_std  = all_feats.std(axis=0, keepdims=True) + 1e-12

    # x_true の統計（教師あり学習の場合のみ）
    if not unsupervised_mode:
        all_xtrue = np.concatenate(
            [rc["x_true_np"] for rc in cases_with_x], axis=0
        )
        x_mean = all_xtrue.mean()
        x_std  = all_xtrue.std() + 1e-12
        log_print(
            f"[INFO] x_true (cases with ground truth): "
            f"min={all_xtrue.min():.3e}, max={all_xtrue.max():.3e}, mean={x_mean:.3e}"
        )
    else:
        # 教師なし学習の場合、ダミー値を設定
        x_mean = 0.0
        x_std  = 1.0
        log_print("[INFO] x_true 統計: 教師なし学習モードのためダミー値 (mean=0, std=1)")

    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t  = torch.tensor(x_std,  dtype=torch.float32, device=device)

    # --- rank ごとの x_true 統計（train ケースのみ） ---
    #     data loss を rank ごとに正規化するための mean/std
    train_ranks = sorted({int(rc["rank"]) for rc in raw_cases_train})
    num_ranks = max(train_ranks) + 1

    sums   = np.zeros(num_ranks, dtype=np.float64)
    sqsums = np.zeros(num_ranks, dtype=np.float64)
    counts = np.zeros(num_ranks, dtype=np.int64)

    # 教師あり学習の場合のみ rank ごと統計を計算
    if not unsupervised_mode:
        for rc in raw_cases_train:
            if not rc.get("has_x_true", False):
                continue
            r = int(rc["rank"])
            x = rc["x_true_np"].astype(np.float64).reshape(-1)
            sums[r]   += x.sum()
            sqsums[r] += np.square(x).sum()
            counts[r] += x.size

    # 初期値としてグローバル mean/std を入れておき、train に存在する rank だけ上書き
    x_mean_rank = np.full(num_ranks, x_mean, dtype=np.float64)
    x_std_rank  = np.full(num_ranks, x_std,  dtype=np.float64)

    for r in range(num_ranks):
        if counts[r] > 0:
            mean_r = sums[r] / counts[r]
            var_r  = sqsums[r] / counts[r] - mean_r * mean_r
            std_r  = np.sqrt(max(var_r, 1e-24))
            x_mean_rank[r] = mean_r
            x_std_rank[r]  = std_r

    log_print("[INFO] rank-wise x_true statistics (train only):")
    for r in range(num_ranks):
        log_print(
            f"  rank={r}: count={counts[r]}, "
            f"mean={x_mean_rank[r]:.3e}, std={x_std_rank[r]:.3e}"
        )

    # torch.Tensor (device 上) として保持
    x_mean_rank_t = torch.from_numpy(x_mean_rank.astype(np.float32)).to(device)
    x_std_rank_t  = torch.from_numpy(x_std_rank.astype(np.float32)).to(device)

    # --- torch ケース化 & w_pde 統計 ---
    # USE_LAZY_LOADING が True の場合、データは CPU に保持され、学習時に GPU へ転送される
    cases_train = []
    cases_val   = []
    w_all_list  = []

    if USE_LAZY_LOADING:
        log_print("[INFO] 遅延GPU転送モード: データはCPUに保持され、使用時のみGPUへ転送されます")

    for rc in raw_cases_train:
        cs = convert_raw_case_to_torch_case(
            rc, feat_mean, feat_std, x_mean, x_std, device,
            lazy_load=USE_LAZY_LOADING
        )
        cases_train.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    for rc in raw_cases_val:
        cs = convert_raw_case_to_torch_case(
            rc, feat_mean, feat_std, x_mean, x_std, device,
            lazy_load=USE_LAZY_LOADING
        )
        cases_val.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    # --- w_pde の分布ログ（全 train+val ケースまとめ） ---
    if w_all_list:
        w_all = np.concatenate(w_all_list, axis=0)

        w_min  = float(w_all.min())
        w_max  = float(w_all.max())
        w_mean = float(w_all.mean())
        p50    = float(np.percentile(w_all, 50))
        p90    = float(np.percentile(w_all, 90))
        p99    = float(np.percentile(w_all, 99))

        log_print("=== w_pde (mesh-quality weights) statistics over all train+val cases ===")
        log_print(f"  count = {w_all.size}")
        log_print(f"  min   = {w_min:.3e}")
        log_print(f"  mean  = {w_mean:.3e}")
        log_print(f"  max   = {w_max:.3e}")
        log_print(f"  p50   = {p50:.3e}")
        log_print(f"  p90   = {p90:.3e}")
        log_print(f"  p99   = {p99:.3e}")
        log_print("==========================================================================")

    num_train = len(cases_train)
    num_val   = len(cases_val)

    # --- モデル定義 ---
    model = SimpleSAGE(
        in_channels=nFeat,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHED_FACTOR,
            patience=LR_SCHED_PATIENCE,
            min_lr=LR_SCHED_MIN_LR,
            verbose=False,
        )

    # --- AMP (混合精度学習) の設定 ---
    use_amp_actual = USE_AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp_actual)
    if use_amp_actual:
        log_print("[INFO] 混合精度学習 (AMP) が有効です")
    else:
        if USE_AMP and device.type != "cuda":
            log_print("[INFO] AMP は CUDA デバイスでのみ有効です。CPU モードでは無効化されます")

    log_print("=== Training start (relative data loss + weighted PDE loss, train/val split) ===")

    # --- 可視化用の準備 ---
    fig, axes = (None, None)
    if enable_plot:
        fig, axes = init_plot()
    history = {
        "epoch": [],
        "loss": [],
        "data_loss": [],
        "pde_loss": [],
        "laplacian_loss": [],  # オートディファレンス損失
        "bc_loss": [],  # WALL_FACES を用いた境界条件損失
        "gauge_loss": [],  # ゲージ損失（教師なし学習時のみ）
        "rel_err_train": [],
        "rel_err_val": [],  # val が無いときは None
    }

    # --- 学習ループ ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        total_data_loss = 0.0
        total_pde_loss  = 0.0
        total_laplacian_loss = torch.tensor(0.0, device=device)
        total_bc_loss = torch.tensor(0.0, device=device)
        total_gauge_loss = 0.0  # ゲージ損失（教師なし学習時の定数モード抑制用）
        sum_rel_err_tr  = 0.0
        sum_R_pred_tr   = 0.0
        sum_rmse_tr     = 0.0
        num_cases_with_x = 0  # データ損失を計算したケース数

        # -------- train で勾配計算 --------
        for cs in cases_train:
            # 遅延ロードの場合、ケースデータを GPU に転送
            if USE_LAZY_LOADING:
                cs_gpu = move_case_to_device(cs, device)
            else:
                cs_gpu = cs

            feats       = cs_gpu["feats"]
            edge_index  = cs_gpu["edge_index"]
            x_true      = cs_gpu["x_true"]  # 教師なし学習の場合は None
            b           = cs_gpu["b"]
            row_ptr     = cs_gpu["row_ptr"]
            col_ind     = cs_gpu["col_ind"]
            vals        = cs_gpu["vals"]
            row_idx     = cs_gpu["row_idx"]
            w_pde       = cs_gpu["w_pde"]
            wall_bc_index = cs_gpu.get("wall_bc_index")
            wall_bc_value = cs_gpu.get("wall_bc_value")
            wall_bc_weight = cs_gpu.get("wall_bc_weight")
            has_x_true  = cs_gpu.get("has_x_true", x_true is not None)

            # ラプラシアン損失用に座標へ勾配を通す（必要なときのみ）
            if USE_AUTODIFF_LAPLACIAN_LOSS:
                feats_for_model = cs_gpu["feats"].detach().clone().requires_grad_(True)
            else:
                feats_for_model = feats

            # AMP: autocast で順伝播と損失計算を FP16/BF16 で実行
            with torch.cuda.amp.autocast(enabled=use_amp_actual):
                # モデルは正規化スケールで出力
                x_pred_norm = model(feats_for_model, edge_index)
                # 非正規化スケールに戻す
                x_pred = x_pred_norm * x_std_t + x_mean_t

                # データ損失: x_true がある場合のみ計算
                if has_x_true and x_true is not None:
                    # rank ごとの mean/std を用いた x の正規化（data loss 用）
                    rank_id = int(cs["rank"])
                    mean_r  = x_mean_rank_t[rank_id]
                    std_r   = x_std_rank_t[rank_id]

                    # x_true, x_pred を rank ごとに標準化
                    x_true_norm_case = (x_true - mean_r) / (std_r + 1e-12)
                    x_pred_norm_case_for_loss = (x_pred - mean_r) / (std_r + 1e-12)

                    # データ損失: rank ごとに正規化した MSE
                    data_loss_case = F.mse_loss(
                        x_pred_norm_case_for_loss,
                        x_true_norm_case
                    )
                    num_cases_with_x += 1
                else:
                    # 教師なし学習: データ損失は 0
                    data_loss_case = torch.tensor(0.0, device=device)

                # PDE 損失: w_pde 付き相対残差²
                Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                r  = Ax - b

                sqrt_w = torch.sqrt(w_pde)
                wr = sqrt_w * r
                wb = sqrt_w * b
                norm_wr = torch.norm(wr)
                norm_wb = torch.norm(wb) + EPS_RES
                R_pred = norm_wr / norm_wb
                pde_loss_case = R_pred * R_pred

                # オートディファレンスによるラプラシアン損失
                if USE_AUTODIFF_LAPLACIAN_LOSS:
                    laplacian_loss_case = compute_autodiff_laplacian_loss(
                        x_pred, feats_for_model[:, 0:3], b, w_pde
                    )
                else:
                    laplacian_loss_case = torch.tensor(0.0, device=device)

                # ゲージ損失: x_pred の平均値の二乗（教師なし学習時の定数モード抑制用）
                # 圧力ポアソン方程式の解は定数の不定性（ゲージ自由度）があるため、
                # 平均ゼロに近づけることで解を一意に定める
                gauge_loss_case = torch.mean(x_pred) ** 2

                # 境界条件損失（WALL_FACES）: Dirichlet を想定し、重み付き MSE
                if USE_BC_LOSS and wall_bc_index is not None and wall_bc_index.numel() > 0:
                    bc_diff = x_pred[wall_bc_index] - wall_bc_value
                    bc_loss_case = torch.mean(wall_bc_weight * bc_diff * bc_diff)
                else:
                    bc_loss_case = torch.tensor(0.0, device=device)

            total_data_loss = total_data_loss + data_loss_case
            total_pde_loss  = total_pde_loss  + pde_loss_case
            total_laplacian_loss = total_laplacian_loss + laplacian_loss_case
            total_bc_loss = total_bc_loss + bc_loss_case
            total_gauge_loss = total_gauge_loss + gauge_loss_case

            with torch.no_grad():
                # rel_err, RMSE: x_true がある場合のみ計算
                if has_x_true and x_true is not None:
                    # ゲージ不変評価: 両者を平均ゼロに正規化してから比較
                    # 圧力ポアソン方程式の解は定数の不定性があるため、
                    # 公平な比較のために平均を引いてから誤差を計算
                    x_pred_centered = x_pred - torch.mean(x_pred)
                    x_true_centered = x_true - torch.mean(x_true)
                    diff = x_pred_centered - x_true_centered
                    N = x_true.shape[0]
                    rel_err_case = torch.norm(diff) / (torch.norm(x_true_centered) + EPS_DATA)
                    rmse_case    = torch.sqrt(torch.sum(diff * diff) / N)
                    sum_rel_err_tr += rel_err_case.item()
                    sum_rmse_tr    += rmse_case.item()
                sum_R_pred_tr  += R_pred.detach().item()

            # 遅延ロードの場合、GPU メモリを解放
            if USE_LAZY_LOADING:
                del cs_gpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # 損失の計算（教師なし学習の場合は PDE 損失 + ゲージ損失）
        total_pde_loss = total_pde_loss / num_train
        if USE_AUTODIFF_LAPLACIAN_LOSS:
            total_laplacian_loss = total_laplacian_loss / num_train
        if USE_BC_LOSS:
            total_bc_loss = total_bc_loss / num_train
        total_gauge_loss = total_gauge_loss / num_train
        laplacian_term = (
            LAMBDA_LAPLACIAN * total_laplacian_loss
            if USE_AUTODIFF_LAPLACIAN_LOSS
            else torch.tensor(0.0, device=device)
        )
        bc_term = (
            LAMBDA_BC * total_bc_loss
            if USE_BC_LOSS
            else torch.tensor(0.0, device=device)
        )
        if unsupervised_mode or num_cases_with_x == 0:
            # 教師なし学習: PDE 損失 + ゲージ正則化
            # ゲージ正則化は圧力ポアソンの定数モード（ゲージ自由度）を抑制
            total_data_loss = torch.tensor(0.0, device=device)
            loss = LAMBDA_PDE * total_pde_loss + laplacian_term + bc_term + LAMBDA_GAUGE * total_gauge_loss
        else:
            # 教師あり学習: データ損失 + PDE 損失（ゲージ正則化は不要、x_true が定数モードを固定）
            total_data_loss = total_data_loss / num_cases_with_x
            loss = LAMBDA_DATA * total_data_loss + LAMBDA_PDE * total_pde_loss + laplacian_term + bc_term

        # AMP: スケーリングされた勾配で逆伝播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        avg_rel_err_val = None
        avg_R_pred_val = None
        avg_rmse_val = None

        # スケジューラ用・ロギング用に検証誤差を計算（必要なときのみ）
        need_val_eval = num_val > 0 and (scheduler is not None or epoch % PLOT_INTERVAL == 0)
        if need_val_eval:
            model.eval()
            sum_rel_err_val = 0.0
            sum_R_pred_val = 0.0
            sum_rmse_val = 0.0
            num_val_with_x = 0  # x_true があるケース数
            with torch.no_grad():
                for cs in cases_val:
                    # 遅延ロードの場合、ケースデータを GPU に転送
                    if USE_LAZY_LOADING:
                        cs_gpu = move_case_to_device(cs, device)
                    else:
                        cs_gpu = cs

                    feats = cs_gpu["feats"]
                    edge_index = cs_gpu["edge_index"]
                    x_true = cs_gpu["x_true"]
                    b = cs_gpu["b"]
                    row_ptr = cs_gpu["row_ptr"]
                    col_ind = cs_gpu["col_ind"]
                    vals = cs_gpu["vals"]
                    row_idx = cs_gpu["row_idx"]
                    w_pde = cs_gpu["w_pde"]
                    has_x_true = cs_gpu.get("has_x_true", x_true is not None)

                    with torch.cuda.amp.autocast(enabled=use_amp_actual):
                        x_pred_norm = model(feats, edge_index)
                        x_pred = x_pred_norm * x_std_t + x_mean_t

                    # rel_err, RMSE: x_true がある場合のみ計算
                    if has_x_true and x_true is not None:
                        # ゲージ不変評価: 両者を平均ゼロに正規化してから比較
                        x_pred_centered = x_pred - torch.mean(x_pred)
                        x_true_centered = x_true - torch.mean(x_true)
                        diff = x_pred_centered - x_true_centered
                        rel_err = torch.norm(diff) / (torch.norm(x_true_centered) + EPS_DATA)
                        N = x_true.shape[0]
                        rmse = torch.sqrt(torch.sum(diff * diff) / N)
                        sum_rel_err_val += rel_err.item()
                        sum_rmse_val += rmse.item()
                        num_val_with_x += 1

                    Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                    r = Ax - b
                    sqrt_w = torch.sqrt(w_pde)
                    wr = sqrt_w * r
                    wb = sqrt_w * b
                    norm_wr = torch.norm(wr)
                    norm_wb = torch.norm(wb) + EPS_RES
                    R_pred = norm_wr / norm_wb

                    sum_R_pred_val += R_pred.item()

                    # 遅延ロードの場合、GPU メモリを解放
                    if USE_LAZY_LOADING:
                        del cs_gpu
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

            avg_R_pred_val = sum_R_pred_val / num_val
            if num_val_with_x > 0:
                avg_rel_err_val = sum_rel_err_val / num_val_with_x
                avg_rmse_val = sum_rmse_val / num_val_with_x
            else:
                # 教師なし学習: PDE 残差を指標として使用
                avg_rel_err_val = avg_R_pred_val
                avg_rmse_val = 0.0

        # 学習率スケジューラを更新（検証誤差があればそれを監視）
        if scheduler is not None:
            metric_for_scheduler = avg_rel_err_val if avg_rel_err_val is not None else loss.item()
            scheduler.step(metric_for_scheduler)


        # --- ロギング（train + val） ---
        if epoch % PLOT_INTERVAL == 0 or epoch == 1:
            # 教師あり学習の場合のみ相対誤差を計算
            if unsupervised_mode or num_cases_with_x == 0:
                avg_rel_err_tr = sum_R_pred_tr / num_train  # PDE 残差を代用
                avg_rmse_tr    = 0.0
            else:
                avg_rel_err_tr = sum_rel_err_tr / num_cases_with_x
                avg_rmse_tr    = sum_rmse_tr / num_cases_with_x
            avg_R_pred_tr  = sum_R_pred_tr / num_train

            current_lr = optimizer.param_groups[0]["lr"]

            avg_rel_err_val = None
            avg_R_pred_val  = None
            avg_rmse_val    = None

            if num_val > 0 and avg_rel_err_val is None:
                # スケジューラを使っていない場合などで、まだ val を計算していないときのみ算出
                model.eval()
                sum_rel_err_val = 0.0
                sum_R_pred_val  = 0.0
                sum_rmse_val    = 0.0
                num_val_with_x = 0
                with torch.no_grad():
                    for cs in cases_val:
                        # 遅延ロードの場合、ケースデータを GPU に転送
                        if USE_LAZY_LOADING:
                            cs_gpu = move_case_to_device(cs, device)
                        else:
                            cs_gpu = cs

                        feats      = cs_gpu["feats"]
                        edge_index = cs_gpu["edge_index"]
                        x_true     = cs_gpu["x_true"]
                        b          = cs_gpu["b"]
                        row_ptr    = cs_gpu["row_ptr"]
                        col_ind    = cs_gpu["col_ind"]
                        vals       = cs_gpu["vals"]
                        row_idx    = cs_gpu["row_idx"]
                        w_pde      = cs_gpu["w_pde"]
                        has_x_true = cs_gpu.get("has_x_true", x_true is not None)

                        with torch.cuda.amp.autocast(enabled=use_amp_actual):
                            x_pred_norm = model(feats, edge_index)
                            x_pred = x_pred_norm * x_std_t + x_mean_t

                        # rel_err, RMSE: x_true がある場合のみ計算
                        if has_x_true and x_true is not None:
                            diff = x_pred - x_true
                            rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                            N = x_true.shape[0]
                            rmse  = torch.sqrt(torch.sum(diff * diff) / N)
                            sum_rel_err_val += rel_err.item()
                            sum_rmse_val    += rmse.item()
                            num_val_with_x += 1

                        Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                        r  = Ax - b
                        sqrt_w = torch.sqrt(w_pde)
                        wr = sqrt_w * r
                        wb = sqrt_w * b
                        norm_wr = torch.norm(wr)
                        norm_wb = torch.norm(wb) + EPS_RES
                        R_pred = norm_wr / norm_wb

                        sum_R_pred_val  += R_pred.item()

                        # 遅延ロードの場合、GPU メモリを解放
                        if USE_LAZY_LOADING:
                            del cs_gpu
                            if device.type == "cuda":
                                torch.cuda.empty_cache()

                avg_R_pred_val = sum_R_pred_val / num_val
                if num_val_with_x > 0:
                    avg_rel_err_val = sum_rel_err_val / num_val_with_x
                    avg_rmse_val    = sum_rmse_val    / num_val_with_x
                else:
                    # 教師なし学習: PDE 残差を指標として使用
                    avg_rel_err_val = avg_R_pred_val
                    avg_rmse_val    = 0.0

            # 履歴に追加
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["data_loss"].append((LAMBDA_DATA * total_data_loss).item())
            history["pde_loss"].append((LAMBDA_PDE * total_pde_loss).item())
            history["laplacian_loss"].append(laplacian_term.item())
            history["bc_loss"].append(bc_term.item())
            history["gauge_loss"].append((LAMBDA_GAUGE * total_gauge_loss).item())
            history["rel_err_train"].append(avg_rel_err_tr)
            history["rel_err_val"].append(avg_rel_err_val)  # None の可能性あり

            # プロット更新
            if enable_plot:
                update_plot(fig, axes, history)

            # コンソールログ
            log = (
                f"[Epoch {epoch:5d}] loss={loss.item():.4e}, "
                f"lr={current_lr:.3e}, "
                f"data_loss={LAMBDA_DATA * total_data_loss:.4e}, "
                f"PDE_loss={LAMBDA_PDE * total_pde_loss:.4e}, "
            )
            if USE_AUTODIFF_LAPLACIAN_LOSS:
                log += f"laplacian_loss={laplacian_term:.4e}, "
            if USE_BC_LOSS:
                log += f"bc_loss={bc_term:.4e}, "
            if unsupervised_mode or num_cases_with_x == 0:
                # 教師なし学習: ゲージ損失も表示
                log += f"gauge_loss={LAMBDA_GAUGE * total_gauge_loss:.4e}, "
            log += (
                f"rel_err_train(avg)={avg_rel_err_tr:.4e}, "
#                f"RMSE_train(avg)={avg_rmse_tr:.4e}, "
#                f"R_pred_train(avg)={avg_R_pred_tr:.4e}"
            )
            if avg_rel_err_val is not None:
                log += (
#                    f", rel_err_val(avg)={avg_rel_err_val:.4e}, "
                    f", rel_err_val(avg)={avg_rel_err_val:.4e} "
#                    f"RMSE_val(avg)={avg_rmse_val:.4e}, "
#                    f"R_pred_val(avg)={avg_R_pred_val:.4e}"
                )
            log_print(log)

    # 学習終了後、インタラクティブモードを解除してウィンドウを保持したい場合はコメントアウト解除
    # plt.ioff()
    # plt.show()

    # --- 最終プロットの保存 ---
    # すべての history を使って最終状態の図を更新・保存
    if enable_plot and len(history["epoch"]) > 0:
        final_plot_filename = (
            f"training_history_"
            f"DATA{lambda_data_tag}_"
            f"PDE{lambda_pde_tag}.png"
        )
        final_plot_path = os.path.join(OUTPUT_DIR, final_plot_filename)

        update_plot(fig, axes, history)
        fig.savefig(final_plot_path, dpi=200, bbox_inches='tight')
        log_print(f"[INFO] Training history figure saved to {final_plot_path}")

    # --- 実行時間の計測結果をログ出力 ---
    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60.0
    log_print(
        f"[INFO] Total elapsed time: {elapsed:.2f} s "
        f"(~{h:02d}:{m:02d}:{s:05.2f})"
    )

    # ログファイルをクローズ
    if LOGGER_FILE is not None:
        LOGGER_FILE.close()
        LOGGER_FILE = None

    # --- 最終評価: OpenFOAM 解との PDE 残差比較を含む ---
    log_print("\n=== Final diagnostics (train cases) ===")
    model.eval()

    # ★ ここでカウンタを初期化（関数のこのスコープ内）
    num_error_plots_train = 0

    for cs in cases_train:
        time_str   = cs["time"]
        rank_str   = cs["rank"]

        # 遅延ロードの場合、ケースデータを GPU に転送
        if USE_LAZY_LOADING:
            cs_gpu = move_case_to_device(cs, device)
        else:
            cs_gpu = cs

        feats      = cs_gpu["feats"]
        edge_index = cs_gpu["edge_index"]
        x_true     = cs_gpu["x_true"]
        b          = cs_gpu["b"]
        row_ptr    = cs_gpu["row_ptr"]
        col_ind    = cs_gpu["col_ind"]
        vals       = cs_gpu["vals"]
        row_idx    = cs_gpu["row_idx"]
        w_pde      = cs_gpu["w_pde"]
        has_x_true = cs_gpu.get("has_x_true", x_true is not None)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp_actual):
                x_pred_norm = model(feats, edge_index)
                x_pred = x_pred_norm * x_std_t + x_mean_t

            # 学習で使った weighted PDE 残差
            Ax_pred_w = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
            r_pred_w  = Ax_pred_w - b
            sqrt_w    = torch.sqrt(w_pde)
            wr_pred   = sqrt_w * r_pred_w
            wb        = sqrt_w * b
            norm_wr   = torch.norm(wr_pred)
            norm_wb   = torch.norm(wb) + EPS_RES
            R_pred_w  = norm_wr / norm_wb

            # 物理的な（非加重）PDE 残差: GNN 解
            Ax_pred = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
            r_pred  = Ax_pred - b
            norm_r_pred    = torch.norm(r_pred)
            max_abs_r_pred = torch.max(torch.abs(r_pred))
            norm_b         = torch.norm(b)
            norm_Ax_pred   = torch.norm(Ax_pred)
            R_pred_over_b  = norm_r_pred / (norm_b + EPS_RES)
            R_pred_over_Ax = norm_r_pred / (norm_Ax_pred + EPS_RES)

            # 教師あり学習の場合のみ x_true との比較
            if has_x_true and x_true is not None:
                diff = x_pred - x_true
                N = x_true.shape[0]
                rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                rmse    = torch.sqrt(torch.sum(diff * diff) / N)

                # 物理的な（非加重）PDE 残差: OpenFOAM 解
                Ax_true = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_true)
                r_true  = Ax_true - b
                norm_r_true    = torch.norm(r_true)
                max_abs_r_true = torch.max(torch.abs(r_true))
                norm_Ax_true   = torch.norm(Ax_true)
                R_true_over_b  = norm_r_true / (norm_b + EPS_RES)
                R_true_over_Ax = norm_r_true / (norm_Ax_true + EPS_RES)

        if has_x_true and x_true is not None:
            log_print(
                f"  [train] Case (time={time_str}, rank={rank_str}): "
                f"rel_err = {rel_err.item():.4e}, RMSE = {rmse.item():.4e}, "
                f"R_pred(weighted) = {R_pred_w.item():.4e}"
            )
            log_print(f"    x_true: min={x_true.min().item():.6e}, max={x_true.max().item():.6e}, "
                  f"mean={x_true.mean().item():.6e}, norm={torch.norm(x_true).item():.6e}")
            log_print(f"    x_pred: min={x_pred.min().item():.6e}, max={x_pred.max().item():.6e}, "
                  f"mean={x_pred.mean().item():.6e}, norm={torch.norm(x_pred).item():.6e}")
            log_print(f"    x_pred_norm: min={x_pred_norm.min().item():.6e}, "
                  f"max={x_pred_norm.max().item():.6e}, mean={x_pred_norm.mean().item():.6e}")
            log_print(f"    diff (x_pred - x_true): norm={torch.norm(diff).item():.6e}")
            log_print(f"    正規化パラメータ: x_mean={x_mean_t.item():.6e}, x_std={x_std_t.item():.6e}")

            log_print("    [PDE residual comparison vs OpenFOAM]")
            log_print(
                "      GNN : "
                f"||r||_2={norm_r_pred.item():.6e}, "
                f"max|r_i|={max_abs_r_pred.item():.6e}, "
                f"||r||/||b||={R_pred_over_b.item():.5f}, "
                f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
            )
            log_print(
                "      OF  : "
                f"||r||_2={norm_r_true.item():.6e}, "
                f"max|r_i|={max_abs_r_true.item():.6e}, "
                f"||r||/||b||={R_true_over_b.item():.5f}, "
                f"||r||/||Ax||={R_true_over_Ax.item():.5f}"
            )

            # --- ここでスケール診断 ---
            a, b_fit, rmse_before, rmse_after = compute_affine_fit(x_true, x_pred)
            log_print(
                f"    [Affine fit x_pred->x_true] "
                f"a={a:.3e}, b={b_fit:.3e}, "
                f"RMSE_before={rmse_before:.3e}, RMSE_after={rmse_after:.3e}, "
                f"RMSE_ratio={rmse_after / rmse_before:.3f}"
            )
        else:
            # 教師なし学習: PDE 残差のみ表示
            log_print(
                f"  [train] Case (time={time_str}, rank={rank_str}) [教師なし学習]: "
                f"R_pred(weighted) = {R_pred_w.item():.4e}"
            )
            log_print(f"    x_pred: min={x_pred.min().item():.6e}, max={x_pred.max().item():.6e}, "
                  f"mean={x_pred.mean().item():.6e}, norm={torch.norm(x_pred).item():.6e}")
            log_print(
                "    [PDE residual (GNN)]"
                f" ||r||_2={norm_r_pred.item():.6e}, "
                f"max|r_i|={max_abs_r_pred.item():.6e}, "
                f"||r||/||b||={R_pred_over_b.item():.5f}, "
                f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
            )

        # 予測結果の書き出し
        x_pred_np = x_pred.cpu().numpy().reshape(-1)
        out_path = os.path.join(OUTPUT_DIR, f"x_pred_train_{time_str}_rank{rank_str}.dat")
        with open(out_path, "w") as f:
            for i, val in enumerate(x_pred_np):
                f.write(f"{i} {val:.9e}\n")
        log_print(f"    [INFO] train x_pred を {out_path} に書き出しました。")

        # ★ 誤差場の可視化（train ケース、x_true がある場合のみ）
        if enable_error_plots and has_x_true and x_true is not None and num_error_plots_train < MAX_ERROR_PLOT_CASES_TRAIN:
            prefix = f"train_time{time_str}_rank{rank_str}"
            save_error_field_plots(cs, x_pred, x_true, prefix)
            num_error_plots_train += 1

        # 遅延ロードの場合、GPU メモリを解放
        if USE_LAZY_LOADING:
            del cs_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if num_val > 0:
        log_print("\n=== Final diagnostics (val cases) ===")

        # ★ val 側のカウンタもここで初期化
        num_error_plots_val = 0

        for cs in cases_val:
            time_str   = cs["time"]
            rank_str   = cs["rank"]

            # 遅延ロードの場合、ケースデータを GPU に転送
            if USE_LAZY_LOADING:
                cs_gpu = move_case_to_device(cs, device)
            else:
                cs_gpu = cs

            feats      = cs_gpu["feats"]
            edge_index = cs_gpu["edge_index"]
            x_true     = cs_gpu["x_true"]
            b          = cs_gpu["b"]
            row_ptr    = cs_gpu["row_ptr"]
            col_ind    = cs_gpu["col_ind"]
            vals       = cs_gpu["vals"]
            row_idx    = cs_gpu["row_idx"]
            w_pde      = cs_gpu["w_pde"]
            has_x_true = cs_gpu.get("has_x_true", x_true is not None)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp_actual):
                    x_pred_norm = model(feats, edge_index)
                    x_pred = x_pred_norm * x_std_t + x_mean_t

                Ax_pred_w = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                r_pred_w  = Ax_pred_w - b
                sqrt_w    = torch.sqrt(w_pde)
                wr_pred   = sqrt_w * r_pred_w
                wb        = sqrt_w * b
                norm_wr   = torch.norm(wr_pred)
                norm_wb   = torch.norm(wb) + EPS_RES
                R_pred_w  = norm_wr / norm_wb

                Ax_pred = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                r_pred  = Ax_pred - b
                norm_r_pred    = torch.norm(r_pred)
                max_abs_r_pred = torch.max(torch.abs(r_pred))
                norm_b         = torch.norm(b)
                norm_Ax_pred   = torch.norm(Ax_pred)
                R_pred_over_b  = norm_r_pred / (norm_b + EPS_RES)
                R_pred_over_Ax = norm_r_pred / (norm_Ax_pred + EPS_RES)

                # 教師あり学習の場合のみ x_true との比較
                if has_x_true and x_true is not None:
                    diff = x_pred - x_true
                    N = x_true.shape[0]
                    rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                    rmse    = torch.sqrt(torch.sum(diff * diff) / N)

                    Ax_true = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_true)
                    r_true  = Ax_true - b
                    norm_r_true    = torch.norm(r_true)
                    max_abs_r_true = torch.max(torch.abs(r_true))
                    norm_Ax_true   = torch.norm(Ax_true)
                    R_true_over_b  = norm_r_true / (norm_b + EPS_RES)
                    R_true_over_Ax = norm_r_true / (norm_Ax_true + EPS_RES)

            if has_x_true and x_true is not None:
                log_print(
                    f"  [val]   Case (time={time_str}, rank={rank_str}): "
                    f"rel_err = {rel_err.item():.4e}, RMSE = {rmse.item():.4e}, "
                    f"R_pred(weighted) = {R_pred_w.item():.4e}"
                )
                log_print(f"    x_true: min={x_true.min().item():.6e}, max={x_true.max().item():.6e}, "
                      f"mean={x_true.mean().item():.6e}, norm={torch.norm(x_true).item():.6e}")
                log_print(f"    x_pred: min={x_pred.min().item():.6e}, max={x_pred.max().item():.6e}, "
                      f"mean={x_pred.mean().item():.6e}, norm={torch.norm(x_pred).item():.6e}")
                log_print(f"    x_pred_norm: min={x_pred_norm.min().item():.6e}, "
                      f"max={x_pred_norm.max().item():.6e}, mean={x_pred_norm.mean().item():.6e}")
                log_print(f"    diff (x_pred - x_true): norm={torch.norm(diff).item():.6e}")
                log_print(f"    正規化パラメータ: x_mean={x_mean_t.item():.6e}, x_std={x_std_t.item():.6e}")

                log_print("    [PDE residual comparison vs OpenFOAM]")
                log_print(
                    "      GNN : "
                    f"||r||_2={norm_r_pred.item():.6e}, "
                    f"max|r_i|={max_abs_r_pred.item():.6e}, "
                    f"||r||/||b||={R_pred_over_b.item():.5f}, "
                    f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
                )
                log_print(
                    "      OF  : "
                    f"||r||_2={norm_r_true.item():.6e}, "
                    f"max|r_i|={max_abs_r_true.item():.6e}, "
                    f"||r||/||b||={R_true_over_b.item():.5f}, "
                    f"||r||/||Ax||={R_true_over_Ax.item():.5f}"
                )

                # --- ここでスケール診断 ---
                a, b_fit, rmse_before, rmse_after = compute_affine_fit(x_true, x_pred)
                log_print(
                    f"    [Affine fit x_pred->x_true] "
                    f"a={a:.3e}, b={b_fit:.3e}, "
                    f"RMSE_before={rmse_before:.3e}, RMSE_after={rmse_after:.3e}, "
                    f"RMSE_ratio={rmse_after / rmse_before:.3f}"
                )
            else:
                # 教師なし学習: PDE 残差のみ表示
                log_print(
                    f"  [val]   Case (time={time_str}, rank={rank_str}) [教師なし学習]: "
                    f"R_pred(weighted) = {R_pred_w.item():.4e}"
                )
                log_print(f"    x_pred: min={x_pred.min().item():.6e}, max={x_pred.max().item():.6e}, "
                      f"mean={x_pred.mean().item():.6e}, norm={torch.norm(x_pred).item():.6e}")
                log_print(
                    "    [PDE residual (GNN)]"
                    f" ||r||_2={norm_r_pred.item():.6e}, "
                    f"max|r_i|={max_abs_r_pred.item():.6e}, "
                    f"||r||/||b||={R_pred_over_b.item():.5f}, "
                    f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
                )

            x_pred_np = x_pred.cpu().numpy().reshape(-1)
            out_path = os.path.join(OUTPUT_DIR, f"x_pred_val_{time_str}_rank{rank_str}.dat")
            with open(out_path, "w") as f:
                for i, val in enumerate(x_pred_np):
                    f.write(f"{i} {val:.9e}\n")
            log_print(f"    [INFO] val x_pred を {out_path} に書き出しました。")

            # ★ 誤差場の可視化（val ケース、x_true がある場合のみ）
            if enable_error_plots and has_x_true and x_true is not None and num_error_plots_val < MAX_ERROR_PLOT_CASES_VAL:
                prefix = f"val_time{time_str}_rank{rank_str}"
                save_error_field_plots(cs, x_pred, x_true, prefix)
                num_error_plots_val += 1

            # 遅延ロードの場合、GPU メモリを解放
            if USE_LAZY_LOADING:
                del cs_gpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if return_history:
        return history

if __name__ == "__main__":
    train_gnn_auto_trainval_pde_weighted(DATA_DIR)

