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
- `WALL_FACES ...` セクション : 境界セル ID と境界値（および任意の重み）。`pEqn_*` 内で検出されると Dirichlet 境界損失に利用されます。

## 実行方法
```bash
python GNN_train_val_weight.py
```
主なハイパーパラメータやフラグはスクリプト冒頭で定義されています（例：`DATA_DIR`, `NUM_EPOCHS`, `LR`）。

Fourier 特徴量を座標に付加する場合は、以下のフラグを指定します（未指定なら従来と同じ入力次元で学習します）。

```bash
python GNN_train_val_weight.py --fourier-features --fourier-k 4
```

追加する Fourier 特徴量は、各周波数 `k = 1..K` について座標ベクトル \(\mathbf{x}\) に対し

\[
\big[\sin(2\pi k \mathbf{x}),\ \cos(2\pi k \mathbf{x})\big]
\]

を生成し既存ノード特徴に連結します（自動で入力次元が拡張されます）。

## 損失関数と物理拘束
ここでは主要な損失を数式でまとめます。`\hat{x}` はモデル出力、`x` は教師データ、`b` は RHS、`A` は疎行列、`w_{pde}` はメッシュ品質ベースの重み、`N` はサンプル数を表します。

- **データ損失**（教師ありのみ）：rank ごとに標準化した MSE を計算し、その平均を用います。
  \[
  \mathcal{L}_{\text{data}} = \frac{1}{N}\sum_{i=1}^N \big\|\text{standardize}(\hat{x}_i) - \text{standardize}(x_i)\big\|_2^2
  \]

- **PDE 損失**：疎行列を使った残差を重み付きで相対化します。
  \[
  r = A\hat{x} - b, \qquad
  \mathcal{L}_{\text{PDE}} = \frac{\|\sqrt{w_{pde}}\, r\|_2^2}{\|\sqrt{w_{pde}}\, b\|_2^2 + \varepsilon}
  \]

- **境界条件損失（WALL_FACES がある場合）**：境界セルの Dirichlet 条件を明示的に重み付き MSE で拘束します。
  \[
  \mathcal{L}_{\text{bc}} = \frac{1}{N_{\text{bc}}}\sum_{i \in \text{WALL\_FACES}} w_i \big(\hat{x}_i - x_i^{\text{bc}}\big)^2
  \]

- **ゲージ損失**（教師なしのみ）：圧力の定数不定性を抑制するために平均をゼロへ近づけます。
  \[
  \mathcal{L}_{\text{gauge}} = \big(\text{mean}(\hat{x})\big)^2
  \]

- **自動微分ラプラシアン損失（オプション）**：座標勾配から直接ラプラシアンを評価し、RHS と比較する相対残差をとります。
  \[
  r_{\Delta} = \nabla^2 \hat{x} - b, \qquad
  \mathcal{L}_{\text{lap}} = \frac{\|\sqrt{w_{pde}}\, r_{\Delta}\|_2^2}{\|\sqrt{w_{pde}}\, b\|_2^2 + \varepsilon}
  \]

総損失は利用モードに応じて、`LAMBDA_*` で重み付けした線形和として組み合わせます。

## 学習・評価フロー
1. `find_time_rank_list` でデータを収集し、`TRAIN_FRACTION` に従って train/val に分割。
2. `USE_LAZY_LOADING` によりデータを CPU 常駐のまま必要時に GPU に転送可能。
3. AMP（`USE_AMP`）を利用して高速化しつつ、データ損失 + PDE 損失 +（任意で）ラプラシアン損失 + 教師なし時のゲージ損失を合算して学習。
4. 検証では教師ありなら相対誤差、教師なしなら PDE 残差を指標としてログ出力し、必要に応じて予測結果を書き出します。

## 補助スクリプト
- `PINN_improvement_proposals.md`：PINN 改善アイデアのメモ。モデル・損失の拡張やコロケーション点戦略などのヒントをまとめています。
- `hyperparameter_search_optuna.py`：Optuna を用いたハイパーパラメータ探索の雛形（必要に応じて利用してください）。
