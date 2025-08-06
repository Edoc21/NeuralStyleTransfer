import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

import os
from glob import glob
from PIL import Image

class ContentLoss(nn.Module):
    """コンテンツ損失を計算するクラス"""
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.clone().detach()

    def forward(self, input_tensor):
        self.loss = F.mse_loss(input_tensor, self.target)
        return input_tensor

def gram_matrix(input_tensor):
    """グラム行列を計算するヘルパー関数"""
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    """スタイル損失を計算するクラス"""
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).clone().detach()

    def forward(self, input_tensor):
        G = gram_matrix(input_tensor)
        self.loss = F.mse_loss(G, self.target)
        return input_tensor

class Normalization(nn.Module):
    """画像を正規化するクラス"""
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class NeuralTransferExe:
    """ニューラルスタイル転送を実行するメインクラス"""
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        self.imsize = args.imgsize if torch.cuda.is_available() else 128
        self.path = args.path
        self.num_steps = args.num_steps
        self.style_weight = args.style_weight
        self.content_weight = args.content_weight
        self.input_use_content = args.input_use_content
        self.content_img_number = args.content_img_number
        self.style_img_number = args.style_img_number

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        self.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.style_img_path = None
        self.content_img_path = None
        self.style_img = None
        self.content_img = None
        self.input_img = None
        self.model = None
        self.style_losses = []
        self.content_losses = []
        self.optimizer = None
    
    def _image_loader(self, image_name):
        """画像をロードして前処理する内部ヘルパーメソッド"""
        loader = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor()
        ])
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def _load_images(self):
        """スタイル画像とコンテンツ画像をロードする"""
        style_img_path_list = glob(os.path.join(self.path, "style", "*"))
        content_img_path_list = glob(os.path.join(self.path, "content", "*"))
        
        if not style_img_path_list or not content_img_path_list:
            raise FileNotFoundError("スタイル画像またはコンテンツ画像が見つかりません。")

        self.style_img_path = style_img_path_list[int(self.style_img_number)]
        self.content_img_path = content_img_path_list[self.content_img_number]
        
        self.style_img = self._image_loader(self.style_img_path)
        self.content_img = self._image_loader(self.content_img_path)

        assert self.style_img.size() == self.content_img.size(), \
            "スタイル画像とコンテンツ画像のサイズが一致しません。"
        
        if self.input_use_content:
            self.input_img = self.content_img.clone().requires_grad_(True)
        else:
            self.input_img = torch.randn_like(self.content_img, requires_grad=True)
            
        self.optimizer = optim.LBFGS([self.input_img])

    def _prepare_model_and_losses(self):
        """VGGモデルを構築し、損失関数を組み込む"""
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std)
        model = nn.Sequential(normalization)
        
        i = 0
        for layer in self.cnn.children():
            name = ""
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                self.content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                self.style_losses.append(style_loss)
                
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        self.model = model[:(i + 1)]

    def _run_optimization(self):
        """最適化ループを実行する"""
        run = [0]
        while run[0] <= self.num_steps:
            def closure():
                with torch.no_grad():
                    self.input_img.clamp_(0, 1)

                self.optimizer.zero_grad()
                self.model(self.input_img)
                
                style_score = sum(sl.loss for sl in self.style_losses)
                content_score = sum(cl.loss for cl in self.content_losses)

                style_score *= self.style_weight
                content_score *= self.content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"run {run[0]}:")
                    print(f'Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}')
                    print()

                return style_score + content_score

            self.optimizer.step(closure)

    def _save_result(self):
        """結果画像を保存する"""
        with torch.no_grad():
            self.input_img.clamp_(0, 1)
        
        unloader = transforms.ToPILImage()
        image = self.input_img.cpu().clone().squeeze(0)
        image = unloader(image)

        content_name = os.path.basename(self.content_img_path)
        style_name = os.path.basename(self.style_img_path)
        output_path = os.path.join(self.path, "output", f"{os.path.splitext(content_name)[0]}_{os.path.splitext(style_name)[0]}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

        print("-" * 50)
        print("Success Transfer!")
        print("Please check output folder!")
        print("-" * 50)

    def run_style_transfer(self):
        """スタイル転送の全体の流れを制御する"""
        print('Loading images and building the model...')
        self._load_images()
        self._prepare_model_and_losses()

        self.model.eval()
        self.model.requires_grad_(False)
        self.input_img.requires_grad_(True)
        
        print('Starting optimization...')
        self._run_optimization()
        self._save_result()

# 実行方法:
# poetry run python -m main --style-img-number "対象のstyle画像numberを入力" --content-img-number "対象のcontent画像numberを入力"
# 実行例）
# poetry run python -m main --style-img-number 0 --content-img-number 1
def main():
    import argparse

    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(
        description="Neural Transferのハイパーパラメータを設定"
    )
    parser.add_argument(
        "--imgsize",
        type=int,
        default=512,
        help="出力画像サイズ(正方形)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./",
        help="ルートパス",
    )
    parser.add_argument(
        "--input-use-content",
        type=bool,
        default=True,
        help="True:入力画像にcontentを使用  False:入力画像に乱数を使用",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=300,
        help="Step回数",
    )
    parser.add_argument(
        "--style-weight",
        type=int,
        default=1000000,
        help="スタイル：コンテント比",
    )
    parser.add_argument(
        "--content-weight",
        type=int,
        default=1,
        help="コンテント重み係数",
    )
    parser.add_argument(
        "--content-img-number",
        type=int,
        default=0,
        help="コンテント画像選択",
    )
    parser.add_argument(
        "--style-img-number",
        type=int,
        default=0,
        help="スタイル画像選択",
    )
    # コマンドライン引数を解析
    args = parser.parse_args()

    transfer = NeuralTransferExe(args)
    # エージェントを実行して最終的な出力を取得
    transfer.run_style_transfer()

if __name__ == "__main__":
    main()