#!/usr/bin/python
# -*- coding:utf-8 -*-
import os


def get_dir_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_model_path():
    return get_dir_path() +"/model"


def get_CosyVoice2_path():
    return get_model_path() +"/CosyVoice2-0.5B"