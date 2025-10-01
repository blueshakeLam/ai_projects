#!/usr/bin/python
# -*- coding:utf-8 -*-
import os.path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import  DataLoader
from torchvision.transforms import functional as F
from os.path import join
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import warnings

from load_image import LoadImage
from sdmatte_config import *

warnings.filterwarnings("ignore")

def load_model(config_dir, model_dir):
    # 将numpy的_reconstruct函数添加到安全全局变量中，防止后续报错[2](@ref)
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

    # 保存原始的 torch.load 函数
    _original_torch_load = torch.load

    # 定义一个新的函数，强制设置 weights_only=False
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    # 用新函数覆盖原函数
    torch.load = patched_torch_load

    # 此后，所有代码（包括第三方库）对 torch.load 的调用都会默认使用 weights_only=False
    # initializing model
    torch.set_grad_enabled(False)

    cfg = LazyConfig.load(config_dir)
    #改为本地路径
    cfg.model.pretrained_model_name_or_path = get_sdmatte_path()

    model = instantiate(cfg.model)
    model.to('cuda')
    DetectionCheckpointer(model).load(checkpoint_dir)

    model.eval()
    return cfg, model

# model and output
def matting_inference(
    config_dir="",
    checkpoint_dir="",
    input_dir="",
    out_dir="",
):
    cfg, model = load_model(config_dir, checkpoint_dir)

    test_dataset = LoadImage(input_dir,psm=cfg.hy_dict.psm, radius=cfg.hy_dict.radius)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)
    #
    # # inferencing
    os.makedirs(out_dir, exist_ok=True)

    for data in tqdm(test_loader):
        image_name = data["image_name"][0]
        H, W = data["hw"][0].item(), data["hw"][1].item()

        with torch.no_grad():
            pred = model(data)
            output = pred.flatten(0, 2) * 255
            output = cv2.resize(output.detach().cpu().numpy(), (W, H)).astype(np.uint8)
            output = F.to_pil_image(output).convert("L")
            image = Image.open(input_dir +"/"+ image_name)
            image.putalpha(output)
            image.save(join(out_dir, image_name.replace(".jpg", ".png")))
            torch.cuda.empty_cache()

if __name__ == "__main__":

    config_dir = get_sdmatte_config_path()
    checkpoint_dir = get_sdmatte_ckpt_path()
    intput_dir = "b:/1"
    out_dir = "b:/2"

    matting_inference(
        config_dir=config_dir,
        checkpoint_dir=checkpoint_dir,
        input_dir=intput_dir,
        out_dir=out_dir,
    )

