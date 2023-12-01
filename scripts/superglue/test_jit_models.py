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
    parser.add_argument("--superglue", "-sg", type=str, required=True)
    parser.add_argument("--image0", "-i0", type=str, required=True)
    parser.add_argument("--image1", "-i1", type=str, required=True)
    parser.add_argument("--match_threshold", "-m", type=float, default=0.2)
    parser.add_argument("--keypoint_threshold", 