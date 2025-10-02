#!/usr/bin/python
# -*- coding:utf-8 -*-
import os


def get_dir_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_model_path():
    return get_dir_path() + "/model"


def get_hunyuan3d_omni_model_path():
    return get_model_path() + "/Hunyuan3D-Omni"


def get_facebook_dinov2_large_path():
    return get_model_path() + "/facebook-dinov2-large"
