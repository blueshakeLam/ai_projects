#!/usr/bin/python
# -*- coding:utf-8 -*-

import os

def get_dir_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_model_path():
    return get_dir_path() + "/model"


def get_spart_tts_path():
    return get_model_path() + "/Spark-TTS-0.5B"

