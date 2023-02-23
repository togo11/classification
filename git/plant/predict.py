import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import models as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.547, 0.58, 0.52], [0.137, 0.123, 0.124])])

    # load image
    img_path = "/data/dwx/dataset/ani_add_split/train/hua/0101_161白桦.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    print_res = "class: {}   prob: {:.3}".format(str("betula"),
                                                 0.94)
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
