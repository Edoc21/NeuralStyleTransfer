# Neural Style Transfer CLI with PyTorch

This repository provides a CLI tool for Neural Style Transfer based on the [official PyTorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).

## 📘 Languages

- 🇯🇵 [日本語はこちら](README.ja.md)

---

This repository provides a simple CLI tool for Neural Style Transfer based on the official PyTorch tutorial.
You can run the style transfer via notebook or directly from the command line.

### 🖌 What is Neural Style Transfer?
Neural Style Transfer is a deep learning technique that combines the content of one image with the style of another, creating a new, artistic image.
It extracts the structure from the content image and textures/colors from the style image to blend them together.

### 🛠 How to Use
#### 1. Prepare Images
Create two folders: content/ and style/.

Place images inside them, named using the format: NN_image_name.jpg (e.g., 00_city.jpg, 01_monet.jpg).

Make sure the images are loadable with PIL.

#### 2. Run
Run with Jupyter notebook:

```bash
コピーする
編集する
python -m NeuralTransferLibrary.main --imgsize 512 --content-img-number 0 --style-img-number 1
```
Or run from the command line:

```bash
コピーする
編集する
python -m NeuralTransferLibrary.main --imgsize 512 --content-img-number 0 --style-img-number 1
```

#### 3. Output
The generated image will be saved in the output/ folder.

### ⚙️ Command-line Arguments
|Argument|Default|Description|
| ---- | ---- | ----|
|--imgsize|512|Output image size (square)|
|--path|"./"|Root path of the project|
|--input-use-content|True|True: use content image as input False: use random noise as input|
|--num-steps|300|Number of optimization steps|
|--style-weight|1000000|Weight for style loss|
|--content-weight|1|Weight for content loss|
|--content-img-number|0|Select content image by number (matches file prefix)|
|--style-img-number|0|Select style image by number (matches file prefix)|

### 📄 License
This project is licensed under the MIT License.