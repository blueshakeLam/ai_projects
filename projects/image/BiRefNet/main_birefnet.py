#!/usr/bin/python
# -*- coding:utf-8 -*-
# Use codes and weights locally
import torch

from models.birefnet import BiRefNet
from birefnet_config import get_birefnet_model_path
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

birefnet = BiRefNet.from_pretrained(get_birefnet_model_path())
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()
birefnet.half()
def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask
# Visualization
plt.axis("off")
plt.imshow(extract_object(birefnet, imagepath="b:/1/1.png")[0])
plt.show()
