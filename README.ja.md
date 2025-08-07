# Neural Style Transfer CLI with PyTorch

## 📌 日本語 (Japanese)

このリポジトリは、[PyTorch公式のNeural Style Transferチュートリアル](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) をベースに、簡単にスタイル転送を実行できるCLIツールを提供します。  
ノートブックでも、コマンドラインでもスタイル転送を手軽に試すことができます。

### 🖌 Neural Style Transferとは？

Neural Style Transfer（ニューラルスタイル転送）は、**ある画像の内容（content）**と**別の画像のスタイル（style）**を融合し、新しい芸術的な画像を生成する技術です。  
コンテンツ画像は形や構造、スタイル画像は色彩や筆致などを提供し、それらをディープラーニングで統合します。

---

### 🛠 使い方

#### 1. 画像の準備

- `content/` フォルダと `style/` フォルダを作成し、それぞれに **「二桁の番号_画像名」** という形式で画像を保存してください。  
  例: `00_city.jpg`, `01_monet.jpg`  
  （画像はPILで読み込み可能な形式にしてください）

#### 2. 実行方法

- ノートブックで実行:
```bash
python -m NeuralTransferLibrary.main --imgsize 512 --content-img-number 0 --style-img-number 1
```
CLIから直接実行:

```bash
コピーする
編集する
python -m NeuralTransferLibrary.main --imgsize 512 --content-img-number 0 --style-img-number 1
```

#### 3. 出力
生成された画像は output/ フォルダに保存されます。

#### ⚙️ 引数一覧
|引数名|デフォルト値|説明|
| ---- | ---- | ----|
|--imgsize|512|出力画像サイズ（縦横同じピクセル数）|
|--path|"./"|プロジェクトのルートパス|
|--input-use-content|True|True: content画像を入力に使用 False:ノイズから生成|
|--num-steps|300|最適化のステップ数|
|--style-weight|1000000|スタイル損失の重み|
|--content-weight|1|コンテンツ損失の重み|
|--content-img-number|0|contentフォルダ内の使用画像（番号指定）|
|--style-img-number|0|styleフォルダ内の使用画像（番号指定）|

#### 📄 ライセンス
このプロジェクトは MITライセンス のもとで公開されています。