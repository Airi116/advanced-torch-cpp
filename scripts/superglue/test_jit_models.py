import os
from typing import List

import cv2
import numpy as np
import torch

from SuperGluePretrainedNetwork.models.utils import read_image


def get_args():
    import argparse

    parser = argparse.ArgumentParser("")
    parser.add_argument("--superpoint", "-sp", type=str, required=True)
    parser.add_argumen