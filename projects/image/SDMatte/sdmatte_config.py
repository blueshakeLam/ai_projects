#!/usr/bin/python
# -*- coding:utf-8 -*-
import os


def get_dir_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_model_path():
    return get_dir_path() + "/model"

def get_sdmatte_path():
    return get_model_path() + "/SDMatte"

def get_inference_path():
    return get_dir_path() + "/infer_output/SDMatte/bbox/AM-2K"

def get_bg_20k_path():
    return get_dir_path() + "/infer_output/SDMatte/bbox/BG-20K"

def get_sdmatte_ckpt_path():
    return get_model_path() + "/SDMatte/SDMatte.pth"


def get_sdmatte_config_path():
    return get_dir_path() + "/configs/SDMatte.py"

def get_lite_sdmatte_config_path():
    return get_dir_path() + "/configs/LiteSDMatte.py"
