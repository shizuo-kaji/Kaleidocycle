[English](README.md) | [日本語](README.ja.md)

# Kaleidocycle Python ツールキット

`kaleidocycle` は [Kaleidocycle リポジトリ](../README.ja.md) の Python
パッケージです．Maple，Mathematica，MATLAB のコードを Python に移植し，関連論文の構成を実装しています．
数値探索，テータ関数・楕円関数による厳密解，可積分変形，剛性解析，3D プリント用の CAD モデル，
ブラウザで動く [Kaleidocycle Studio](web/README.md)（英語）を含みます．

N 個の四面体からなるカライドサイクルは，`R^3` の `N + 1` 本の単位ヒンジベクトル（従法線，binormal）で表します．
隣り合うヒンジは一定の捩率角 `mu` をなすので，内積はすべて等しくなります．
最後のヒンジが最初と同じ向きに戻るものを *oriented*，逆向きになるもの（`oriented=False`）を
*anti-oriented* と呼びます．後者はメビウスの帯のトポロジーを持ちます．

## 目次

- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [ノートブック](#ノートブック)
- [Kaleidocycle Studio](#kaleidocycle-studio)
- [共有サンプル](#共有サンプル)
- [パッケージ構成](#パッケージ構成)
- [基本プロパティ](#基本プロパティ)
- [厳密解の構成](#厳密解の構成)
  - [テータ関数による構成](#テータ関数による構成)
  - [楕円関数による臨界族](#楕円関数による臨界族)
  - [直角円とトーラス結び目](#直角円とトーラス結び目)
- [可積分変形](#可積分変形)
- [解析](#解析)
  - [レポートと停留性](#レポートと停留性)
  - [局所自由度](#局所自由度)
  - [アニメーションの診断量](#アニメーションの診断量)
- [3Dプリント用CADモデル](#3dプリント用cadモデル)
  - [ヒンジの種類](#ヒンジの種類)
  - [接触チェック](#接触チェック)
- [開発](#開発)
- [その他のドキュメント](#その他のドキュメント)
- [参考文献](#参考文献)

## インストール

Python 3.11 以上が必要です．このディレクトリで次を実行します．

```bash
python -m pip install -e ".[dev]"
```

[uv](https://docs.astral.sh/uv/) を使う場合は次のとおりです．

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

zsh が角括弧をグロブとして解釈しないよう，extras は引用符で囲んでください．

| Extra | 追加されるもの |
| --- | --- |
| `dev` | pytest，Ruff，Hypothesis，JupyterLab（ipywidgets，ipympl），JAX |
| `jax` | JAX のみ．自動微分を使う小さめの実行環境向け |
| `cad` | build123d．[3D プリント用 CAD モデル](#3dプリント用cadモデル)に必要 |

テストは `pytest` で実行します（[開発](#開発)を参照）．

## クイックスタート

```python
from kaleidocycle import Kaleidocycle, export_json, load_sample, principal_family

# ランダムな初期値から，閉じた anti-oriented な K7 を探索する．
kc = Kaleidocycle(7, oriented=False, seed=42)
print(kc.is_closed, kc.constant_torsion)  # True 1.2709...
print(kc.report())                        # 幾何・トポロジー・残差のまとめ
kc.plot()

# 楕円関数の主族による，捩率角の厳密な最小解．
exact = Kaleidocycle(hinges=principal_family(7).binormals(), oriented=False)

# Kaleidocycle Studio と共有の組み込みサンプルを読み込み，JSON に書き出す．
cycle = load_sample("classic_k6_right_angle")
export_json(cycle, "classic_k6.json")
```

ノートブックでは `%matplotlib widget` を使うと 3D プロットを回転できます．

## ノートブック

対話的な例は [`notebooks/`](notebooks/) にあります．3D の操作には `%matplotlib widget`
を使い，ノートブックのカーネルがこのプロジェクトの環境を指していることを確認してください．

| ノートブック | 内容 |
| --- | --- |
| [`FindKaleidocycles`](notebooks/FindKaleidocycles.ipynb) | 最適化による構成． |
| [`ExplicitConstructionWithTheta`](notebooks/ExplicitConstructionWithTheta.ipynb) | テータ関数による解析解と，楕円関数族との比較． |
| [`AlternatingTwistedK6`](notebooks/AlternatingTwistedK6.ipynb) | 隣り合う辺の余弦が `−cos α` と `cos α` を交互にとる，ねじれた紙のリンクからなる K6．明示的な構成と数値解を求め，3 回対称性を確かめます． |
| [`IntegrableDeformations`](notebooks/IntegrableDeformations.ipynb) | 正の mKdV 階層，そのスペクトル・ハミルトニアンと持ち上げ，反周期的な sine–Gordon フロー，保存量の数値的な確認． |
| [`IntegrableTheorems`](notebooks/IntegrableTheorems.ipynb) | 論文 *Integrable deformations of Kaleidocycles* の定理 A–E を対話的にたどります．次元の数え上げと階数の低下，可換なフロー，QRT 曲線，楕円関数族の捩率の指数，大域的最小解と近くのファイバー，連続極限． |
| [`SelfCrossing`](notebooks/SelfCrossing.ipynb) | 第 1 mKdV フローで K12 が自己交差することの精度保証付きの証明．静的な診断と，時刻スライダー付きの 3D 表示． |
| [`KaleidocycleProperties`](notebooks/KaleidocycleProperties.ipynb) | 構造的な性質と残差． |
| [`LocalDoF`](notebooks/LocalDoF.ipynb) | 与えられた配置で拘束を保つ運動の，無限小自由度と有限自由度． |
| [`Animation`](notebooks/Animation.ipynb) | アニメーション用のユーティリティ． |
| [`AnimationScalarProperties`](notebooks/AnimationScalarProperties.ipynb) | アニメーションの診断量の計算とプロット． |
| [`BackendComparison`](notebooks/BackendComparison.ipynb) | NumPy と（オプションの）JAX のソルバー・バックエンドの比較． |
| [`CADModelGeneration`](notebooks/CADModelGeneration.ipynb) | オプションの `build123d` による，3D プリント用のねじれ板（twisted plate）リンクのパラメトリック CAD 出力（`.stl`，`.step`，`.3mf`）． |

## Kaleidocycle Studio

**[Kaleidocycle Studio をブラウザで開く](https://shizuo-kaji.github.io/Kaleidocycle/)**

[`web/`](web/) には，可積分階層によるアニメーションと，閉じた定捩率カライドサイクルの拘束付き編集のための
three.js スタジオがあります．Model lab では楕円関数の臨界円，直角円，トーラス結び目のリングを
ブラウザ上で厳密に構成でき，Analysis カードには曲率の分布，QRT 位相平面，閉じ角 `Theta(k)` のグラフを表示します．
HTTP で配信すると，組み込みメニューはノートブックが `load_sample()` で読むのと同じ
`data/kaleidocycles/` の JSON ファイルを読み込みます．

ローカルで動かすには，次を実行してから <http://localhost:8000/web/> を開きます．

```bash
python -m http.server 8000   # このディレクトリで実行
```

`web/index.html` を直接開いても動きます．ブラウザは `file://` のページから隣の JSON
ファイルを読み込めないため，その場合は生成済みの静的なフォールバックを使います．
数学的な背景と操作方法は [`web/README.md`](web/README.md)（英語）を参照してください．

組み込みメニューで **Self-crossing K12 · first mKdV flow** を選ぶと，接触直前の反例
（`counterexample_k12_self_crossing`）を読み込めます．証明に使う 10 進数の入力は
[`data/counterexamples/`](data/counterexamples/README.md) に，証明と対話的な診断は
`notebooks/SelfCrossing.ipynb` にあります．共有サンプルとオフライン用のフォールバックは
`python scripts/build_self_crossing.py` で再生成できます．

## 共有サンプル

名前付きの配置は `data/kaleidocycles/` にまとめてあり，ノートブックとスタジオの両方がこのカタログを読みます．
`load_sample(name)` で 1 つを読み込み，`sample_catalogue()` でカタログ全体を取得し，
`export_sample()` でノートブックの結果をカタログに追加できます．

```python
from kaleidocycle import export_sample, load_sample

cycle = load_sample("generic_k15_noncritical")

# ノートブックの結果を組み込みカタログに追加する．
export_sample(
    cycle,
    "my_named_sample",
    metadata={"kind": "generic", "non_critical": True},
)
```

`export_json()` は最上位に `name` も書き出します．明示的な `name=...` が優先され，
次に `Kaleidocycle.name`，最後に出力ファイル名が使われます．サンプルを追加・変更したら
`python scripts/build_web_samples.py` を実行し，`catalog.json` と，ファイルを直接開くときだけ使う
生成済みのフォールバックを更新してください．厳密解の注目サンプルは
`python scripts/build_featured_samples.py` で再生成します
（[直角円とトーラス結び目](#直角円とトーラス結び目)を参照）．

## パッケージ構成

モジュールはすべて `src/kaleidocycle/` にあります．可積分変形の関数は
[API リファレンス](api.md)（英語）に表でまとめています．

- [`geometry.py`](src/kaleidocycle/geometry.py): ヒンジのフレーム，曲線，曲率，捩率，ライズ，ツイスト，
  絡み数，曲線・接ベクトル・従法線の相互変換，`Kaleidocycle` クラス．
- [`animation.py`](src/kaleidocycle/animation.py): sine–Gordon フロー，ステップ／ランダムなアニメーション生成，
  フレームの整理・位置合わせ・並べ替え，曲線への変換，`KaleidocycleAnimation`．
- [`integrable.py`](src/kaleidocycle/integrable.py): ねじれ境界条件つきの曲率座標，離散 Frenet
  方程式による再構成，正の mKdV 階層と両立する持ち上げ，twisted Floquet ハミルトニアン，
  反周期的な sine–Gordon 場，Poisson 作用素，変分漸化式，QRT 不変量，時間積分．
- [`elliptic.py`](src/kaleidocycle/elliptic.py): Jacobi の楕円関数による捩率臨界族
  `kappa_n = A cn(omega + n delta; k)`，捩率の分枝と閉じ角，閉じる母数，主族と最小捩率 `mu_N`，
  直角円，トーラス結び目のリング，Jacobi 作用素，捩率のヘッセ行列（指数定理），連続極限の定数．
- [`theta.py`](src/kaleidocycle/theta.py): Jacobi のテータ関数，閉条件の方程式，解析的な曲線，
  テータ関数による従法線とアニメーション．
- [`transform.py`](src/kaleidocycle/transform.py): 鏡映，対合，ねじれの反転，細分，逆順．
- [`constraints.py`](src/kaleidocycle/constraints.py): 単位長，閉条件，端の整列，定捩率，アンカー，
  曲率漸化式の残差．
- [`energies.py`](src/kaleidocycle/energies.py): 曲げ，捩れ，双極子のエネルギー．
- [`solvers.py`](src/kaleidocycle/solvers.py): 最適化の振り分け，Newton 法，擬似逆行列，
  複数のシードからの探索，曲率・捩率のソルバー．
- [`optimality.py`](src/kaleidocycle/optimality.py): 勾配，拘束のヤコビ行列，接空間への射影，停留性の判定．
- [`backends/`](src/kaleidocycle/backends/): 勾配とヤコビ行列を計算する NumPy（有限差分）と
  JAX（自動微分）のバックエンド．
- [`rigidity.py`](src/kaleidocycle/rigidity.py): 剛性行列による診断，無限小機構と自己応力の基底，
  Calladine/Pellegrino の数え上げ，Connelly の 2 次の応力判定，有限運動の追跡．
- [`collisions.py`](src/kaleidocycle/collisions.py)，
  [`collision_certificate.py`](src/kaleidocycle/collision_certificate.py)，
  [`collision_visualization.py`](src/kaleidocycle/collision_visualization.py):
  第 1 mKdV フローでの K12 の自己交差，その区間演算による証明と図．
- [`visualization.py`](src/kaleidocycle/visualization.py): 中心線，ヒンジ，帯，四面体，折り紙の展開図，
  エネルギー，頂点ごとの値のプロット．
- [`cad.py`](src/kaleidocycle/cad.py): オプションの `build123d` による，一体造形（print-in-place）
  できるねじれ板リンクと組立体のパラメトリック 3D CAD 生成．
  [`cad_check.py`](src/kaleidocycle/cad_check.py) は部品の接触を検査します．
- [`io.py`](src/kaleidocycle/io.py): JSON と CSV の入出力．
- [`samples.py`](src/kaleidocycle/samples.py): 名前付きサンプルのカタログ，カタログへの追加，
  見栄えのよい一般サンプルの選択．
- [`report.py`](src/kaleidocycle/report.py): 診断レポートの整形．

## 基本プロパティ

`Kaleidocycle` と `KaleidocycleAnimation` は同じ高水準のチェックを提供します．

| プロパティ | 意味 |
| --- | --- |
| `n` | 四面体の個数．`len(hinges) - 1` に等しい． |
| `oriented` | 最初と最後のヒンジの向きが一致するかどうか． |
| `is_closed` | 接ベクトルの閉条件の残差が許容誤差未満かどうか． |
| `is_aligned` | 両端のヒンジが整列条件を満たすかどうか． |
| `is_unit_norm` | すべてのヒンジベクトルが単位長かどうか． |
| `constant_torsion` | 共通の捩率角（ラジアン）．一定でなければ `None`． |

対応する低水準の残差は `kaleidocycle.constraints` の `closure_residual`，`alignment_residuals`，
`unit_norm_residuals`，`constant_torsion_residuals`，`constraint_residuals`，
`constraint_penalty` で得られます．

## 厳密解の構成

### テータ関数による構成

`generate_theta_curve()` と `generate_theta_binormals()` は，`Kaleidocycle.m`
にあるテータ関数による明示的な構成を実装しています．従法線は Mathematica の `Bx`，`By`，`Bz`
の式に従って計算します．参照式が `G = 0` の付近で除去可能な特異点に近づくパラメータでは，
曲線から符号を推定する方法に切り替えず，同じ式を高い精度で自動的に計算し直します．

```python
from kaleidocycle import generate_theta_binormals

v = 0.07227972073349694
r = 0.30353311936556515
y = 0.9155431292909612
binormals = generate_theta_binormals(v, 0.0, r, y, N=38, t=0.0)
```

### 楕円関数による臨界族

捩率の臨界配置は，整数の巻き数 `m` を持つ Jacobi の楕円関数族です．各族は明示的な母数で閉じ，
`m = 3` の主族は anti-oriented なリングの捩率角の大域的最小値 `mu_N` を実現します．

```python
from kaleidocycle import (
    closing_moduli,
    continuum_constants,
    elliptic_family,
    principal_family,
    torsion_hessian,
)

family = principal_family(9)          # m = 3，閉じる母数 k_9
print(family.torsion_angle)           # mu_9 = 0.945665469...
hinges = family.binormals(phase=0.3)  # 閉じた anti-oriented リング，shape (10, 3)

# 指数定理: 回転を除いた Hess(mu) の慣性指数は (N-m-3, m-3, 1)．
for k in closing_moduli(13, 5):
    saddle = elliptic_family(13, 5, k)
    print(torsion_hessian(saddle.binormals(), sign=saddle.sign).inertia)

print(continuum_constants())          # (k_inf, C_inf) = (0.63092..., 8.09349...)
```

### 直角円とトーラス結び目

論文にあるほかの 2 つの厳密な構成も使えます．`right_angle_family(N, j)` は直角の臨界円
（`c = 0`，`kappa[n+2] = -kappa[n]`，`m = N/2`，`k = sin(2 pi j / N)`）で，
`N = 6, j = 1` は古典的な六角形のカライドサイクルです．`torus_knot_binormals(cells, turns)` は
`N = 3 * cells` 個のヒンジからなる oriented なリングの明示解で，中心線は `(turns, cells)`
トーラス結び目，自己絡み数 `cells * turns` は `N` の 2 乗のオーダーで増えます．
`mirror=True` で鏡映双対をとります．

```python
from kaleidocycle import Kaleidocycle, right_angle_family, torus_knot_binormals

k6 = right_angle_family(6, 1)         # 古典的な六角形のカライドサイクル
print(k6.torsion_angle)               # pi / 2
knot = Kaleidocycle(hinges=torus_knot_binormals(5, 2), oriented=True)  # SL = 10
```

`python scripts/build_featured_samples.py` は，これらの関数から注目サンプル（古典的な K6，
主族の K7 と K48，楕円関数族の鞍点とトーラス結び目，直角円のエネルギー最小解，`SL = 10` の K15）
を再生成します．書き出す前に，閉条件，定捩率，自己絡み数を検証します．

## 可積分変形

```python
import numpy as np

from kaleidocycle import (
    framed_polygon_from_binormals,
    import_json,
    integrate_curvature_flow,
)

kc = import_json("notebooks/output/kaleidocycle_k10_nonoriented_bending.json")
initial = framed_polygon_from_binormals(kc.hinges)
evolution = integrate_curvature_flow(
    initial.curvatures,
    initial.torsion_angle,
    np.linspace(0.0, 1.0, 101),
    flow="sine-gordon",
    sign=initial.sign,
    initial_frame=initial.frames[0],
)

configurations = evolution.configurations()
print(np.ptp(evolution.first_hamiltonian))
print(np.linalg.norm(configurations[-1].closure_residual))
```

可積分変形の API は Cayley 曲率 `kappa = 2*tan(phi/2)` を使います．`sign=1`
は周期的（oriented），`sign=-1` は反周期的（anti-oriented）な従法線を表します．
正の mKdV フローはすべて `flow="mkdvN"`（または `flow=N`）でどちらの符号でも使えます．
sine–Gordon フローは `sign=-1` のときだけ定義されます．インターフェースの全体は
[可積分変形の API](api.md#integrable-deformation-api)（英語）にまとめています．

## 解析

### レポートと停留性

`kc.report()` は幾何，トポロジー，エネルギー，拘束の診断を簡潔にまとめます．
`kc.is_stationary(energy, config=...)` はエネルギーの勾配を拘束の接空間に射影し，
配置が拘束付きの臨界点かどうかを判定します．

```python
from kaleidocycle import Kaleidocycle
from kaleidocycle.constraints import ConstraintConfig

kc = Kaleidocycle(8, oriented=True, seed=42)
config = ConstraintConfig(oriented=True, constant_torsion=True)

print(kc.report())
print(kc.is_stationary("bending", tolerance=1e-4, config=config))
```

### 局所自由度

与えられた配置で拘束を保つ独立な運動がいくつあるかを，相補的な複数の方法で解析できます．

ヤコビ行列を最も精度よく求めるには，使える場合はオプションの JAX バックエンドを使ってください．

```python
BACKEND = "jax"
```

`kc.local_dof()` は *線形*（無限小）の解析を行い，拘束のヤコビ行列 `J` の核 `ker(J)`
の次元を返します．全体の剛体回転 `δh_i = ω × h_i` は常にこの核に含まれるので，
既定ではこれを除きます（`subtract_rigid=True`）．返り値の辞書には `dof`，`raw_dof`，
`rigid_dof`，`rank`，特異値と，必要なら形が `(n+1, 3, dof)` の正規直交基底 `basis` が入ります．

`Kaleidocycle.local_dof()` など接空間を使うメソッドは `Kaleidocycle.analysis_config`
を使い，最後のヒンジを 3 成分すべての整列で拘束します．正則な閉じた配置での既定の自由度は
設計空間の次元 `N - 5` です．`reference_torsion` を指定すると `N - 6` になり，
捩率臨界配置では `N - 5` になります．

`rigidity_svd(hinges, config)` は同じヤコビ行列を剛性行列として扱い，無限小機構（`ker J`）と
自己応力（`ker J.T`）の基底を明示的に返します．`calladine_summary()` は Pellegrino/Calladine
の関係式 `mechanisms - self_stresses = variables - constraints` を，剛体回転を除くかどうかを選んで報告します．
`connelly_second_order_test()` は応力で重み付けたヘッセ行列 `d²(w·F)` を無限小機構の上で評価し，
2 次の障害を調べます．

`kc.finite_motion_dof()` は予測子・修正子法による数値接続で *非線形*（有限運動）の解析を行います．
無限小の核から接方向を選んで有限の予測ステップを進め，Newton 法で拘束多様体に射影し戻し，
得られた変位行列の数値的な階数を報告します．拘束多様体の特異点では，一部の接方向が有限の道に
積分できないため `finite_dof < infinitesimal_dof` となります．

`kc.find_nearby_stationary(energy="mean_cos", config=...)` は `Π_{ker J} ∇E = 0` を
`scipy.optimize.root` で直接解き，拘束の下でのエネルギーの最も近い臨界点を求めます．
勾配降下法と違って鞍点にも収束するので，極小ではなく鞍点になっている解析的な構成
（テータ関数解など）には欠かせません．

`kc.follow_motion(config=..., direction_index=0)` は，選んだ接方向に沿って予測子・修正子法の
数値接続を両方向に 1 回行い，ヒンジのフレームの配列を返します．

`plot_vertex_trajectories(frames)` と `trajectory_dimensionality(curves)`
（`kaleidocycle.visualization`）は運動を可視化・定量化します．補助関数
`align_first_three(curve)` は最初の 3 頂点を固定して剛体運動の不定性を除きます．

Kutzbach–Grübler 式の自由度の数え上げ（一般に `dof = N − 6`）には
`ConstraintConfig(full_alignment=True, reference_torsion=h_0·h_1)` を使ってください．
既定のスカラーの整列条件は，多様体の上では階数に寄与しません．

```python
from kaleidocycle import (
    Kaleidocycle,
    ConstraintConfig,
    calladine_summary,
    connelly_second_order_test,
    rigidity_svd,
    trace_finite_motion,
)

kc = Kaleidocycle(8, oriented=True, seed=42)
BACKEND = "jax"

# Kutzbach-Grübler の「物理的な」拘束系
cfg = ConstraintConfig(
    oriented=True, constant_torsion=True, full_alignment=True,
    reference_torsion=float(kc.hinges[0] @ kc.hinges[1]),
)

rig = rigidity_svd(kc.hinges, cfg, tol=1e-8, backend=BACKEND)
summary = calladine_summary(kc.hinges, cfg, tol=1e-8, backend=BACKEND)

lin = kc.local_dof(config=cfg, tol=1e-6, backend=BACKEND)
print(f"infinitesimal DoF: {lin['dof']}")
print(f"self stresses:     {summary.self_stresses}")

if rig.self_stress_count:
    second = connelly_second_order_test(
        kc.hinges, cfg, stress_index=0, backend=BACKEND
    )
    print(second.eigenvalues)

fin = kc.finite_motion_dof(
    config=cfg, seed=0, step_size=1e-4, n_steps=10, backend=BACKEND
)
print(f"finite DoF:        {fin['finite_dof']}")

frames = trace_finite_motion(
    kc.hinges, cfg, direction_index=0, step_size=5e-4,
    n_steps=80, backend=BACKEND,
)
```

[`notebooks/LocalDoF.ipynb`](notebooks/LocalDoF.ipynb) には，K-G の数え上げ，剛性行列の階数，
Calladine/Pellegrino の自己応力の診断，Connelly の 2 次の判定，メビウス型・テータ関数解の停留性の判定，
自由度 1 の運動での頂点の軌跡の可視化まで，一通りの例があります．このノートブックでは，
現在の API が対応する限り，ヤコビ行列を使う計算に JAX バックエンドを使っています．

### アニメーションの診断量

`KaleidocycleAnimation.compute_scalar_property()` は次のスカラー量を組み込みで計算できます．

| 名前 | 説明 |
| --- | --- |
| `bending_energy` または `bending` | 接ベクトルの Bobenko–Suris 曲げエネルギー． |
| `mean_torsion` | ヒンジ全体での捩率角の平均． |
| `mean_curvature` | ヒンジ全体での離散曲率の平均． |
| `penalty` | 拘束の残差の 2 乗和． |
| `linking_number` | トポロジカルな絡み数． |

独自のスカラー量は関数で追加できます．

```python
def max_curvature(hinges):
    from kaleidocycle.geometry import binormals_to_tangents, pairwise_curvature

    tangents = binormals_to_tangents(hinges, normalize=True)
    curvature = pairwise_curvature(hinges, tangents)
    return float(abs(curvature).max())

anim.compute_scalar_property("max_curvature", func=max_curvature)
```

## 3Dプリント用CADモデル

ねじれ板リンクのパラメトリック 3D CAD 出力（`.stl`，`.step`，`.3mf`）には，オプションの
`build123d` を使います．

```bash
python -m pip install -e ".[cad]"
```

`build123d` がすでに conda 環境に入っていれば，追加のインストールは不要です．

```python
from kaleidocycle import (
    Kaleidocycle,
    check_kaleidocycle_clearance,
    export_print_plate,
    export_twisted_plate,
)

# 捩率角，帯の幅，辺の長さを指定して 1 つのリンクを出力する
export_twisted_plate(
    "unit.stl",
    torsion_angle=72.8,
    ribbon_width=43.0,
    edge_length=29.0,
    degrees=True,
    with_ring=False,  # True にすると一体造形のリングも含める
)

# Kaleidocycle から捩率角を直接とることもできる
kc = Kaleidocycle(7, oriented=False, seed=42)
# Kaleidocycle から .3mf を出力すると，N 個のユニットを 1 枚のプレートに並べる
kc.export_twisted_plate("k7_plate.3mf", ribbon_width=43.0, edge_length=29.0, with_ring=True)
kc.export_twisted_plate("k7_plate_cone.3mf", hinge_type="cone")  # スナップ式のヒンジ
export_print_plate("k7_plate_no_ring.3mf", kc, ribbon_width=43.0, edge_length=29.0)
kc.export_twisted_plate("k7_assembly.step", assembly=True)

# 1 回転分の接触チェック（上のどの出力でも自動で実行される）
print(check_kaleidocycle_clearance(kc, ribbon_width=43.0, edge_length=29.0).summary())
```

`.3mf` の出力は，スライサーのプロジェクト設定を含まない素の 3MF（コア仕様）です．そのため
Bambu Studio，OrcaSlicer，PrusaSlicer はジオメトリとして開き，現在選んでいるプリンタ，
フィラメント，プロセスのプリセットをそのまま使います．各ユニットは 1 つのオブジェクトで
（`with_ring=True` のときは `plate` と `ring` のパーツを持ち），ベッドに接地した状態で
`bed_center=(128, 128)`（256 mm ベッドの中心．小さいプリンタでは別の値を指定）の周りに並びます．
Bambu Studio 02.08.02 は，Bambu のプロジェクト設定を持たない 3MF に対して
"3mf file has invalid config, load geometry data only" という通知を出します．これは 02.08.04
で修正されており，どちらのバージョンでもジオメトリは読み込まれます．

### ヒンジの種類

2 種類のヒンジ（`hinge_type`）で，NovemberKou さんによる印刷用デザインを再現しています．

- `"pin"`（既定，"Material-saving Design"）: 隣り合うリンクを八角形のヒンジピン
  （`build_hinge_pin`）でつなぎます．プリントプレートにはユニットごとに 1 本のピン
  （オブジェクト `hinge_pin`，寝かせた状態．省くには `with_pins=False`）が含まれ，
  組立体ではピンが所定の位置に表示されます．ピンは角から角までが 4 mm（`pin_diameter`）で，
  直径 1.8 mm の中空の芯と両端の 0.5 mm の面取りを持ちます．下側のナックルは 4.0 mm の穴で
  ピンを保持し，角での隙間はゼロ（`pin_fit_clearance=0.0`）なので，ピンは圧入されてがたつきません．
  隣のリンクの上側のナックルは 4.4 mm の穴でピンの周りを回ります（`pin_clearance=0.2`）．
  ナックルは直径 7 mm，軸方向の隙間は 0.2 mm（`hinge_gap`）で，穴は上部が平らな涙滴形なので
  サポートなしで印刷できます．穴がきつく仕上がるプリンタでは `pin_fit_clearance` を少し
  （例えば 0.05 に）上げ，ヒンジが固い場合は `pin_clearance` を上げてください．
- `"cone"`（"Twisted Plate with Ring"）: 別部品はありません．外側の 2 つの下側ナックルに
  底面の直径 4 mm，60 度の円錐（`cone_diameter`，`cone_angle`）があり，隣のリンクの上側ナックルの
  外端にあるソケットにはまります．軸方向の遊びは 0.2 mm（円錐面の法線方向に 0.17 mm）です．
  ナックルは直径 6 mm で，ソケット側のナックルは内側に 2.25 mm の隙間（`snap_gap`）を持ち，
  組み立てるときに円錐の先端（約 0.95 mm）を乗り越えてたわめるようになっています．

既定の帯の幅は 43 mm です．上の `Kaleidocycle(7, oriented=False)` で辺の長さを 29 mm にすると，
幅が 44–45 mm では 2 つ隣のリンクどうしがリングの中心付近で擦れますが，43 mm なら 1 回転を通して
少なくとも 0.9 mm 離れています．

### 接触チェック

どの出力でも，まず部品の接触を検査します（`kaleidocycle.cad_check`）．`check_hinge_fit`
はリンクをその複製とつなぎ，ピンの意図した圧入を除いて，ナックル，ピン，円錐，ソケットが
離れていることを確認します．カライドサイクルの場合は，`check_kaleidocycle_clearance` がさらに
可積分な mKdV／sine–Gordon フローで 1 回転を追跡し，すべてのリンクの組の距離を測ります．
ヒンジを共有しないリンクどうしは `min_clearance`（0.4 mm）以上離れている必要があり，
隣り合うリンクはヒンジ以外で接してはいけません．接触があると `ClearanceWarning`
（`check_clearance="error"` なら `ClearanceError`）を出します．`check_clearance=False`
でチェックを省略できます．一体造形のリングは印刷時の位置で検査します．

## 開発

- `pytest` でテストを実行します．build123d や JAX が必要なテストには `cad`，`jax`
  のマーカーが付いており，`pytest -m "not cad and not jax"` で除外できます．
- `ruff format` と `ruff check` で，1 行 88 文字の PEP 8 に揃えます．その他の規約は
  [`AGENTS.md`](AGENTS.md)（英語）にあります．
- `node web/test_sim.js` は，スタジオの数値計算を Python パッケージと照合します．

| スクリプト | 用途 |
| --- | --- |
| [`build_web_samples.py`](scripts/build_web_samples.py) | `data/kaleidocycles/` を変更した後に `catalog.json` と `web/sample-fallback.js` を再生成． |
| [`build_featured_samples.py`](scripts/build_featured_samples.py) | 厳密解の注目サンプルとフォールバックを再生成． |
| [`build_self_crossing.py`](scripts/build_self_crossing.py) | `data/counterexamples/` から自己交差のサンプルを再生成． |

## その他のドキュメント

- [リポジトリの概要](../README.ja.md): 背景，論文，特許，実物のモデル．
- [API リファレンス](api.md)（英語）: 可積分変形の API と，旧 Mathematica，Maple，MATLAB
  ルーチンの一覧．
- [Kaleidocycle Studio](web/README.md)（英語）: ブラウザ版スタジオの数学的背景と操作方法．
- [自己交差の証明データ](data/counterexamples/README.md)（英語）: 精度保証付きの K12
  自己交差のデータ形式．

## 参考文献

- Shizuo Kaji, Kenji Kajiwara, Shota Shigetomi,
  [*An explicit construction of Kaleidocycles by elliptic theta functions*](https://doi.org/10.1111/sapm.70224),
  Studies in Applied Mathematics 156(5) (2026), e70224,
  [arXiv:2308.04977](https://arxiv.org/abs/2308.04977): `theta.py` のテータ関数による構成．
- Shizuo Kaji, Kenji Kajiwara, Hyeongki Park,
  [*Linkage Mechanisms Governed by Integrable Deformations of Discrete Space Curves*](https://arxiv.org/abs/1903.06360),
  in *Nonlinear Systems and Their Remarkable Mathematical Structures*, Vol. 2,
  CRC Press, 2019: 半離散 mKdV 方程式・sine–Gordon 方程式による運動．
- *Integrable deformations of Kaleidocycles*（原稿）: `integrable.py`，`elliptic.py`，
  `IntegrableTheorems.ipynb` の基礎となる階層，楕円関数族，定理．

論文や解説記事の一覧は [リポジトリの README](../README.ja.md#参考資料) を参照してください．
