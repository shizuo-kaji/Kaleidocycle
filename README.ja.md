[English](README.md) | [日本語](README.ja.md) 

カライドサイクルの設計と動きの解析
==================
カライドサイクルという折り紙は，６つの合同な四面体が数珠繋ぎに輪をなして連なったもので，
イルカのバブルリングのようにクルクルと無限に回すことができます．四面体の個数を増やすこともできますが，たわみやすく動きが不安定になり，綺麗に回すことが難しくなります．
ところが不思議なことに，７つ以上の場合でも特別に計算された四面体を使うと，たわむことなく回ることが発見されました．
この新しいカライドサイクルは，裏表のない帯であるメビウスの帯のトポロジーを持つため，「メビウス・カライドサイクル」と名付けられました．

ここでは，メビウス・カライドサイクルの設計や動きを計算するためのコンピュータ・コードや，
折り紙で作成するための展開図，3Dプリンタで出力するためのモデルファイルを公開しています．

この研究はJSTさきがけ(JPMJPR16E3)の支援を受けて行われました．

# ファイル

- [Kaleidocycle.mw](Kaleidocycle.mw) は Maple のコードで，様々な条件下でカライドサイクルの形状や動きを計算し，可視化します．折り紙展開図の出力機能もあります．
- [Kaleidocycle.nb](Kaleidocycle.nb) は Mathematica のコードで，上の Maple コードの機能限定版です．
- [*3d_model* ディレクトリ](3d_model/) には3Dプリンタで出力可能なモデルがあります.
- [*paper_model* ディレクトリ](paper_model/) には折り紙の展開図があります.
- [*hexagon* directory](hexagon/) にはカライドサイクルの仲間であるヘクサフレクサゴンの折り紙があります．また，カライドサイクルの折り方の説明図もあります．

# 参考資料

## 特許
メビウス・カライドサイクルの形状は以下の特許出願がされていますが，個人利用は自由です．
* Shizuo Kaji, Johannes Schoenke, Eliot Fried, Michael Grunwald, [Moebius Kaleidocycle](https://patentscope2.wipo.int/search/en/detail.jsf?docId=WO2019167941), 特顔2018-033395(Japan), 2019JP007314(PCT), WO 2019167941(Publication Number),2018年2月27日出願

## ビデオ
* [ショートクリップ](https://youtu.be/NULt0lnuVFU)
* [講演動画](https://www.youtube.com/watch?v=0vrXri2z-4w), 折り紙の科学を基盤とするアート・数理 および工学への応用Ⅱ, MIMS「現象数理学研究拠点」共同研究集会, 2021年12月2日
* [オープンキャンパス](https://youtu.be/feZ5x4LjJBc)

## 和文解説
メビウス・カライドサイクルについての和文解説は以下があります．
* 鍛冶静雄, [数理のクロスロード／かたちと動きの数理基盤／(1) リンク万華鏡](https://www.math.kyoto-u.ac.jp/~kaji/papers/susemi201906-linkage.pdf), 数学セミナー 2019年6月号, 日本評論社, 2019.
* 鍛冶静雄, [かたちを算する／おもちゃのかたち](https://www.nippyo.co.jp/shop/magazine/8418.html), 数学セミナー 2021年1月号, 日本評論社, 2021.
* 鍛冶静雄, [曲線の幾何学から生まれた閉リンク機構](https://www.math.kyoto-u.ac.jp/~kaji/papers/linkage.pdf), 2018年度精密工学会春季大会 シンポジウム資料集, pp. 62--65, 2018年3月1日.

## 論文や講演
より学術的な記述は以下にあります．
* 論文, Shizuo Kaji, Kenji Kajiwara, Hyeongki Park, 
[Linkage Mechanisms Governed by Integrable Deformations of Discrete Space Curves](https://arxiv.org/abs/1903.06360), in Nonlinear Systems and Their Remarkable Mathematical Structures, Volume 2, pp 356--381, CRC Press, 2019
* 講演スライド(日本語) [Kaleidocycle](https://www.math.kyoto-u.ac.jp/~kaji/papers/ShapeDesign.pdf), 13 Mar. 2021
* 講演スライド(英語) [Geometry of Kaleidocycles](https://www.math.kyoto-u.ac.jp/~kaji/papers/Kaleidocycle21.pdf), presented at the Kyushu-Illinois Strategic Partnership Colloquia Series #2 Mathematics Without Borders - Applied and Applicable, 11 Mar. 2021
* 古い講演資料 Shizuo Kaji, Eliot Fried, Michael Grunwald, Johannes Schoenke, 
[Geometry of closed kinematic chain](https://www.math.kyoto-u.ac.jp/~kaji/files/Kaleidocycle17.pdf),
IMI Workshop Mathematics in Interface, Dislocation and Structure of Crystals, Nishijin plaza, Fukuoka, 29 Aug. 2017

特許や上記資料のいくつかの内容は，以下の論文にも含まれています．
* Johannes Schoenke and Eliot Fried,
[Single degree of freedom everting ring linkages with nonorientable topology](https://www.pnas.org/content/116/1/90.abstract), PNAS 116 (1), 90--95, 2019.

# ギャラリー

![K9](https://github.com/shizuo-kaji/Kaleidocycle/blob/master/image/K9.gif?raw=true)
![K8](https://github.com/shizuo-kaji/Kaleidocycle/blob/master/image/k8_t.gif?raw=true)
![K15](https://github.com/shizuo-kaji/Kaleidocycle/blob/master/image/K15_link.gif?raw=true)
![K24div](https://github.com/shizuo-kaji/Kaleidocycle/blob/master/image/K24div-trefoil.gif?raw=true)
<img src="https://github.com/shizuo-kaji/Kaleidocycle/blob/master/3d_model/Kaleidocycle_N7Trefoil_all_connected.png?raw=true" width="20%" />
<img src="https://github.com/shizuo-kaji/Kaleidocycle/blob/master/image/3dprint_N12.jpg?raw=true" width="30%" />
<img src="https://github.com/shizuo-kaji/Kaleidocycle/blob/master/image/3dprint-K8.jpg?raw=true" width="30%" />
<img src="https://github.com/shizuo-kaji/Kaleidocycle/blob/master/image/straw-K8.jpg?raw=true" width="30%" />
<img src="https://github.com/shizuo-kaji/Kaleidocycle/blob/master/image/K12_print.jpg?raw=true" width="30%" />


