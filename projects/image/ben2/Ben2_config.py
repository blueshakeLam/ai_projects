#!/usr/bin/python
# -*- coding:utf-8 -*-
import os


def get_dir_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_model_path():
    return get_dir_path() + "/model"


def get_ben2_checkpoints_path():
    return get_model_path() + "/ben2/BEN2_Base.pth"
