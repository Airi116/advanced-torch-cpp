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
    parser.add_argument("--keypoint_threshold", "-k", type=float, default=0.2)
    parser.add_argument("--remove_borders", "-r", type=int, default=4)
    parser.add_argument("--nms_radius", "-n", type=int, default=2)

    return parser.parse_args()


def main(args):
    image_paths = [args.image0, args.image1]

    superpoint = torch.jit.load(args.superpoint)
    superglue = torch.jit.load(args.superglue)

    input_data = [
        read_image(
            image_path, device="cpu", resize=[640, 480], rotation=0, resize_float=False
        )
        for image_path in image_paths
    ]

    images = [input[0] for input in input_data]
    for image in images:
        assert image is not None
    inps = [input[1] for input in input_data]

    batch_inp = torch.cat(inps, dim=0)
    batch_result = superpoint(
        {
            "image": batch_inp,
            "keypoi