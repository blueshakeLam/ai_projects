#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import cv2
import json
import random
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms
from scipy.ndimage import label
import scipy.ndimage

from projects.image.SDMatte.sdmatte_config import get_bg_20k_path

cv2.setNumThreads(1)

class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, sample):
        alpha = sample["alpha"]
        trimap = sample["trimap"]

        if trimap is not None:
            size = alpha.shape[::-1]
            trimap = cv2.resize(trimap, size, interpolation=cv2.INTER_NEAREST)
        else:
            ### generate trimap from alpha
            fg_width = np.random.randint(15, 30)
            bg_width = np.random.randint(15, 30)
            fg_mask = alpha + 1e-5
            bg_mask = 1 - alpha + 1e-5
            fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
            bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

            trimap = np.ones_like(alpha) * 0.5
            trimap[fg_mask > 0.95] = 1.0
            trimap[bg_mask > 0.95] = 0.0

        sample["trimap"] = trimap.astype(np.float32)

        return sample


class GenMask(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]
        self.max_kernel_size = 30
        self.min_kernel_size = 15

    def __call__(self, sample):
        alpha = sample["alpha"]
        mask = sample["mask"]
        h, w = alpha.shape
        if mask is not None:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            ### generate mask
            low = 0.01
            high = 1.0
            thres = random.random() * (high - low) + low
            seg_mask = (alpha >= thres).astype(np.int_).astype(np.uint8)
            random_num = random.randint(0, 3)
            if random_num == 0:
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
            elif random_num == 1:
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
            elif random_num == 2:
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
            elif random_num == 3:
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])

            mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        coords = np.nonzero(mask)
        if coords[0].size == 0 or coords[1].size == 0:
            mask_coords = np.array([0, 0, 1, 1])
        else:
            y_min, x_min = np.argwhere(mask).min(axis=0)
            y_max, x_max = np.argwhere(mask).max(axis=0)
            y_min, y_max = y_min / h, y_max / h
            x_min, x_max = x_min / w, x_max / w
            mask_coords = np.array([x_min, y_min, x_max, y_max])

        sample["mask"] = mask.astype(np.float32)
        sample["mask_coords"] = mask_coords

        return sample


class GenBBox(object):
    def __init__(self, coe_scale=0):
        self.coe_scale = coe_scale

    def __call__(self, sample):
        alpha = sample["alpha"]
        height, width = alpha.shape

        coe = random.uniform(0, self.coe_scale)
        coords = np.nonzero(alpha)

        if coords[0].size == 0 or coords[1].size == 0:
            sample["bbox_mask"], sample["bbox_coords"] = np.zeros_like(alpha).astype(np.float32), np.array([0, 0, 1, 1])
        else:
            binary_mask = alpha > 0
            labeled_array, num_features = label(binary_mask)
            y_min, x_min = np.argwhere(binary_mask).min(axis=0)
            y_max, x_max = np.argwhere(binary_mask).max(axis=0)
            if num_features > 0:
                component_coords = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
                areas = [coords.shape[0] for coords in component_coords]

                sorted_areas_idx = np.argsort(areas)[::-1]
                max_area_idx = sorted_areas_idx[0]
                second_max_area_idx = sorted_areas_idx[1] if len(sorted_areas_idx) > 1 else None

                max_area = areas[max_area_idx]
                second_max_area = areas[second_max_area_idx] if second_max_area_idx is not None else 0

                if max_area >= 10 * second_max_area:
                    max_coords = component_coords[max_area_idx]
                    y_min, x_min = max_coords.min(axis=0)
                    y_max, x_max = max_coords.max(axis=0)

            # Calculate padding_y and padding_x
            padding_y = int(coe * (y_max - y_min))
            padding_x = int(coe * (x_max - x_min))

            # Randomly decide whether to add or subtract padding
            y_min_padding = padding_y if random.choice([True, False]) else -padding_y
            y_max_padding = padding_y if random.choice([True, False]) else -padding_y
            x_min_padding = padding_x if random.choice([True, False]) else -padding_x
            x_max_padding = padding_x if random.choice([True, False]) else -padding_x

            # Apply the padding and ensure it does not exceed the image boundaries
            y_min, y_max = max(0, y_min + y_min_padding), min(height, y_max + y_max_padding)
            x_min, x_max = max(0, x_min + x_min_padding), min(width, x_max + x_max_padding)

            # Generate the bounding box mask
            bbox_mask = np.zeros_like(alpha)
            bbox_mask[y_min:y_max, x_min:x_max] = 1

            y_min, y_max = y_min / height, y_max / height
            x_min, x_max = x_min / width, x_max / width

            # Update the sample dictionary with the bounding box mask and coordinates
            sample["bbox_mask"], sample["bbox_coords"] = bbox_mask.astype(np.float32), np.array([x_min, y_min, x_max, y_max])

        return sample


class GenPoint(object):
    def __init__(self, thres=0, psm="gauss", radius=20):
        self.thres = thres
        self.psm = psm
        self.radius = radius

    def __call__(self, sample):
        alpha = sample["alpha"]
        height, width = alpha.shape
        radius = self.radius

        alpha_mask = (alpha > self.thres).astype(np.float32)
        y_coords, x_coords = np.where(alpha_mask == 1)

        num_points = 10

        if len(y_coords) < num_points:
            sample["point_mask"], sample["point_coords"] = np.zeros_like(alpha).astype(np.float32), np.zeros(20, dtype=np.float32)
            return sample

        selected_indices = np.random.choice(len(y_coords), size=num_points, replace=False)

        point_mask = np.zeros_like(alpha, dtype=np.float32)
        point_coords = []

        for idx in selected_indices:
            y_center = y_coords[idx]
            x_center = x_coords[idx]

            if self.psm == "gauss":
                tmp_mask = np.zeros_like(alpha, dtype=np.float32)
                tmp_mask[y_center, x_center] = 1
                tmp_mask = scipy.ndimage.gaussian_filter(tmp_mask, sigma=radius)
                tmp_mask /= np.max(tmp_mask)
            elif self.psm == "circle":
                tmp_mask = np.zeros_like(alpha, dtype=np.float32)
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        if i**2 + j**2 <= radius**2 and 0 <= x_center + i < alpha.shape[0] and 0 <= y_center + j < alpha.shape[1]:
                            tmp_mask[y_center + j, x_center + i] = 1

            point_mask = np.maximum(point_mask, tmp_mask)

            y_norm = y_center / height
            x_norm = x_center / width
            point_coords.append(x_norm)
            point_coords.append(y_norm)
        if len(point_coords) < 20:
            point_coords = np.concatenate([point_coords, np.zeros(20 - len(point_coords))])

        sample["point_mask"] = point_mask.astype(np.float32)
        sample["point_coords"] = np.array(point_coords[:20])

        return sample


class Gen_Add_Mask_Coord(object):
    def __call__(self, sample):
        trimap = sample["trimap"]

        sample["auto_mask"] = np.ones_like(trimap).astype(np.float32)
        sample["auto_coords"] = np.array([0, 0, 1, 1])
        sample["trimap_coords"] = np.array([0, 0, 1, 1])
        return sample


class CutMask(object):
    def __init__(self, perturb_prob=0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            return sample

        mask = sample["mask"]  # H x W, trimap 0--255, segmask 0--1, alpha 0--1
        h, w = mask.shape
        perturb_size_h, perturb_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
        x = random.randint(0, h - perturb_size_h)
        y = random.randint(0, w - perturb_size_w)
        x1 = random.randint(0, h - perturb_size_h)
        y1 = random.randint(0, w - perturb_size_w)

        mask[x : x + perturb_size_h, y : y + perturb_size_w] = mask[x1 : x1 + perturb_size_h, y1 : y1 + perturb_size_w]

        sample["mask"] = mask
        return sample


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, alpha = sample["image"], sample["alpha"]

        ### resize
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, self.size, interpolation=cv2.INTER_LINEAR)

        sample["alpha"] = alpha
        sample["image"] = image

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(512, 512)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2

    def __call__(self, sample):
        image, alpha, trimap = (
            sample["image"],
            sample["alpha"],
            sample["trimap"],
        )
        h, w = trimap.shape
        if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                image = cv2.resize(
                    image,
                    (int(w * ratio), int(h * ratio)),
                    interpolation=cv2.INTER_LINEAR,
                )
                alpha = cv2.resize(
                    alpha,
                    (int(w * ratio), int(h * ratio)),
                    interpolation=cv2.INTER_LINEAR,
                )
                trimap = cv2.resize(trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(
            zip(*np.where(small_trimap[self.margin // 4 : (h - self.margin) // 4, self.margin // 4 : (w - self.margin) // 4] == 0.5))
        )
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (
                np.random.randint(0, h - self.output_size[0] + 1),
                np.random.randint(0, w - self.output_size[1] + 1),
            )
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0] * 4, unknown_list[idx][1] * 4)

        image_crop = image[
            left_top[0] : left_top[0] + self.output_size[0],
            left_top[1] : left_top[1] + self.output_size[1],
            :,
        ]
        alpha_crop = alpha[left_top[0] : left_top[0] + self.output_size[0], left_top[1] : left_top[1] + self.output_size[1]]
        trimap_crop = trimap[left_top[0] : left_top[0] + self.output_size[0], left_top[1] : left_top[1] + self.output_size[1]]

        if len(np.where(trimap == 0.5)[0]) == 0:
            image_crop = cv2.resize(image, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)

        sample.update({"image": image_crop, "alpha": alpha_crop, "trimap": trimap_crop})
        return sample


class RandomGray(object):
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        image = sample["image"]
        if random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.stack([image] * 3, axis=-1)
        sample["image"] = image

        return sample


class Normalize:
    """Normalize image values by first mapping from [0, 255] to [0, 1] and then
    applying standardization.
    """

    def normalize_img(self, img):
        assert img.dtype == np.float32
        scaled = img.copy() * 2 - 1
        return scaled

    def __call__(self, sample):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        sample["image"] = self.normalize_img(sample["image"])
        sample["alpha"] = self.normalize_img(sample["alpha"])
        sample["trimap"] = self.normalize_img(sample["trimap"])
        sample["mask"] = self.normalize_img(sample["mask"])
        sample["bbox_mask"] = self.normalize_img(sample["bbox_mask"])
        sample["point_mask"] = self.normalize_img(sample["point_mask"])
        sample["auto_mask"] = self.normalize_img(sample["auto_mask"])
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """

    def __call__(self, sample):
        image, alpha, is_trans = sample["image"], sample["alpha"], sample["is_trans"]

        sample["image"], sample["alpha"], sample["is_trans"] = (
            F.to_tensor(image).float(),
            F.to_tensor(alpha).float(),
            torch.tensor(is_trans).long(),
        )
        if "trimap" in sample and "trimap_coords" in sample:
            sample["trimap"], sample["trimap_coords"] = (
                F.to_tensor(sample["trimap"]).float(),
                torch.from_numpy(sample["trimap_coords"]).float(),
            )
        if "mask" in sample and "mask_coords" in sample:
            sample["mask"], sample["mask_coords"] = (
                F.to_tensor(sample["mask"]).float(),
                torch.from_numpy(sample["mask_coords"]).float(),
            )
        if "bbox_mask" in sample and "bbox_coords" in sample:
            sample["bbox_mask"], sample["bbox_coords"] = (
                F.to_tensor(sample["bbox_mask"]).float(),
                torch.from_numpy(sample["bbox_coords"]).float(),
            )
        if "point_mask" in sample and "point_coords" in sample:
            sample["point_mask"], sample["point_coords"] = (
                F.to_tensor(sample["point_mask"]).float(),
                torch.from_numpy(sample["point_coords"]).float(),
            )
        if "auto_mask" in sample and "auto_coords" in sample:
            sample["auto_mask"], sample["auto_coords"] = (
                F.to_tensor(sample["auto_mask"]).float(),
                torch.from_numpy(sample["auto_coords"]).float(),
            )
        return sample
class LoadImage(Dataset):
    def __init__(self, set_list, psm="gauss", radius=20):

        self.samples = self.get_data_list(set_list)

        self.transform = transforms.Compose([
            Resize((1024, 1024)),
            GenTrimap(),
            GenMask(),
            GenBBox(),
            GenPoint(0.8, psm, radius + 10),
            Gen_Add_Mask_Coord(),
            Normalize(),
            ToTensor(),
        ])

        self.fg_num = len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx % self.fg_num]
        alpha = cv2.imread(data["alpha"], 0).astype(np.float32) / 255.0
        H, W = alpha.shape
        image_name = os.path.split(data["alpha"])[-1]

        if "image" in data:
            image = cv2.imread(data["image"], 1).astype(np.float32) / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            bg = cv2.imread(data["bg"], 1)
            fg = cv2.imread(data["fg"], 1)
            image, alpha = composition(fg, bg, alpha)
            image = image.astype(np.float32) / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if "trimap" in data:
            trimap = cv2.imread(data["trimap"], 0).astype(np.float32)
            trimap[trimap == 128] = 0.5
            trimap[trimap == 255] = 1.0
        else:
            trimap = None

        if "mask" in data:
            mask = cv2.imread(data["mask"], 0).astype(np.float32)
            mask[mask == 255] = 1.0
        else:
            mask = None
        if "caption" in data:
            caption = data["caption"]
        else:
            caption = ""
        if "is_trans" in data:
            is_trans = data["is_trans"]
        else:
            raise ValueError("There is no is_trans in the sample.")
        sample = {
            "image": image,
            "alpha": alpha,
            "trimap": trimap,
            "mask": mask,
            "is_trans": is_trans,
            "image_name": image_name,
            "caption": caption,
            "hw": (H, W),
        }

        assert alpha.shape == image.shape[:2]
        assert image.shape[-1] == 3

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.fg_num


    def get_data_list(self, image_dir):

        samples = []
        label_dir = image_dir
        trimap_dir = image_dir

        label_list = os.listdir(label_dir)

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            trimap_path = os.path.join(trimap_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["trimap"] = trimap_path
            path["is_trans"] = 0
            samples.append(path)
        return samples

def resize_bg(bg, alpha):
    if bg.shape[0] > alpha.shape[0] and bg.shape[1] > alpha.shape[1]:
        random_h = random.randint(0, bg.shape[0] - alpha.shape[0])
        random_w = random.randint(0, bg.shape[1] - alpha.shape[1])
        bg = bg[random_h : random_h + alpha.shape[0], random_w : random_w + alpha.shape[1], :]
    else:
        bg = cv2.resize(bg, (alpha.shape[1], alpha.shape[0]), cv2.INTER_LINEAR)
    return bg


def composition(fg, bg, alpha):
    ori_alpha = alpha.copy()
    h, w = alpha.shape
    if random.random() < 1:
        bg = cv2.resize(bg, (2 * w, h), cv2.INTER_LINEAR)
        alpha = np.concatenate((alpha, alpha), axis=1)
        fg = np.concatenate((fg, fg), axis=1)
    else:
        bg = cv2.resize(bg, (w, h), cv2.INTER_LINEAR)
    alpha[alpha < 0] = 0
    alpha[alpha > 1] = 1
    fg[fg < 0] = 0
    fg[fg > 255] = 255
    bg[bg < 0] = 0
    bg[bg > 255] = 255
    if random.random() < 0.5:
        rand_kernel = random.choice([20, 30, 40, 50, 60])
        bg = cv2.blur(bg, (rand_kernel, rand_kernel))
    image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
    if ori_alpha.shape != alpha.shape:
        if random.random() < 0.5:
            alpha[:, :w] = 0
        else:
            alpha[:, w:] = 0
    return image, alpha