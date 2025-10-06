#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import BEN2
from PIL import Image

from Ben2_config import *


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open("b:/2.png") # your image here
    model = BEN2.BEN_Base().to(device).eval()  # init pipeline

    model.loadcheckpoints(get_ben2_checkpoints_path())

    foreground = model.inference(image)
    foreground.save("b:/bg_2.png")

if __name__ == "__main__":
    main()
