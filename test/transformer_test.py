import sys
import os
sys.path.insert(1, os.getcwd())

"""
Usage:
    python3 test/transformer_test.py
"""

import torch
import torch.nn as nn

from spatial_transformer.transformer import SpatialTransformer
from utils import *


def test_identity_transform(input, transformer):
    """
    Output image should be identical to input image when 
    theta is initialized to identity transform.
    """
    B, H, W, C = input.shape

    # identity transform
    # theta = [1., 0., 0., 0., 1., 0.]
    theta = torch.tensor([[1., 0, 0], [0, 1., 0]], dtype=torch.float32).flatten()

    # define loc net weight and bias
    loc_in = H * W * C
    loc_out = 6
    W_loc = nn.Parameter(torch.zeros(loc_in, loc_out), requires_grad=True)
    b_loc = nn.Parameter(theta, requires_grad=True)

    # tie everything together
    # fc_loc = [[1., 0., 0., 0., 1., 0.],
    #           [1., 0., 0., 0., 1., 0.],
    #           [1., 0., 0., 0., 1., 0.],
    #           [1., 0., 0., 0., 1., 0.]]
    fc_loc = torch.matmul(torch.zeros(B, loc_in), W_loc) + b_loc

    # apply spatial transformer
    output = transformer.forward(input, fc_loc)
    print("Identity Transformed Image Shape: {}".format(output.shape))

    tensor2img(output[0]).show()


def test_rotation(input, transformer):
    B, H, W, C = input.shape

    # initialize affine transform tensor `theta`
    degree = 45
    theta = torch.tensor([
        [torch.cos(torch.tensor(deg2rad(degree))), -torch.sin(torch.tensor(deg2rad(degree))), 0],
        [torch.sin(torch.tensor(deg2rad(degree))), torch.cos(torch.tensor(deg2rad(degree))), 0]
    ], dtype=torch.float32).flatten()

    # define loc net weight and bias
    loc_in = H * W * C
    loc_out = 6
    W_loc = nn.Parameter(torch.zeros(loc_in, loc_out), requires_grad=True)
    b_loc = nn.Parameter(theta, requires_grad=True)

    # tie everything together
    fc_loc = torch.matmul(torch.zeros(B, loc_in), W_loc) + b_loc

    # apply spatial transformer
    output = transformer.forward(input, fc_loc)
    print("Rotated Image Shape: {}".format(output.shape))

    tensor2img(output[0]).show()


if __name__ == "__main__":
    DIMS = (600, 600)  # (H, W)
    input = load_images('data/cats/', DIMS)
    # (4, H, W, 3)
    print("Input Image Shape: {}".format(input.shape))

    transformer = SpatialTransformer()

    test_identity_transform(input, transformer)
    test_rotation(input, transformer)