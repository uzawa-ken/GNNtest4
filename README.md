# GNN PINN Trainer

このリポジトリは、圧力ポアソン方程式を対象にしたグラフニューラルネットワーク（GNN）の学習スクリプト `GNN_train_val_weight.py` と、PINN 改善アイデアのメモをまとめた `PINN_improvement_proposals.md` を含みます。メッシュ品質に基づく PDE 残差、ゲージ正則化、オプションで自動微分によるラプラシアン損失を組み合わせ、教師あり・教師なし双方で利用できる構成になっています。

## 依存ライブラリ
- Python 3.8+
- PyTorch
- PyTorch Geometric (`torch_geometric`)
- NumPy, Matplotlib

CUDA を利用する場合は、対応する PyTorch / PyTorch Geometric をインストールしてください。

## データ配置
`data/processor*/gnn/` 以下から `(time, rank)` を自動検出して学習/検証を行います。最低限必要なファイルは次のとおりです。
- `pEqn_{time}_rank{rank}.dat` : スカラー場 RHS
- `A_csr_{time}.dat` または `A_csr_{time}_rank{rank}.dat` : PDE の疎行列
- `x_{time}_rank{rank}.dat` : 教師あり学習時の真値（教師なしの場合は省略可）

## 実行方法
```bash
python GNN_train_val_weight.py
```
主なハイパーパラメータやフラグはスクリプト冒頭で定義されています（例：`DATA_DIR`, `NUM_EPOCHS`, `LR`）。

## 損失関数と物理拘束
- **データ損失**：教師ありモード時に rank ごとの標準化 MSE を計算（`LAMBDA_DATA`）。
- **PDE 損失**：メッシュ品質に基づく重み `w_pde` を用いた相対残差²（`LAMBDA_PDE`）。
- **ゲージ損失**：教師なし学習時に解の定数モードを抑える平均値ペナルティ（`LAMBDA_GAUGE`）。
- **自動微分ラプラシアン損失（オプション）**：座標勾配から直接ラプラシアンを計算し、RHS と比較する相対残差²（`USE_AUTODIFF_LAPLACIAN_LOSS`, `LAMBDA_LAPLACIAN`）。

## 学習・評価フロー
1. `find_time_rank_list` でデータを収集し、`TRAIN_FRACTION` に従って train/val に分割。
2. `USE_LAZY_LOADING` によりデータを CPU 常駐のまま必要時に GPU に転送可能。
3. AMP（`USE_AMP`）を利用して高速化しつつ、データ損失 + PDE 損失 +（任意で）ラプラシアン損失 + 教師なし時のゲージ損失を合算して学習。
4. 検証では教師ありなら相対誤差、教師なしなら PDE 残差を指標としてログ出力し、必要に応じて予測結果を書き出します。

## 補助スクリプト
- `PINN_improvement_proposals.md`：PINN 改善アイデアのメモ。モデル・損失の拡張やコロケーション点戦略などのヒントをまとめています。
- `hyperparameter_search_optuna.py`：Optuna を用いたハイパーパラメータ探索の雛形（必要に応じて利用してください）。
