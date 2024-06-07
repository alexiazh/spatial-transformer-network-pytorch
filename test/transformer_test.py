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


def test_affine_grid_generator_identity(transformer):
    h, w = 5, 5

    # identity transform
    theta = torch.tensor(
        [[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32)

    batch_grids = transformer.affine_grid_generator(h, w, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    x_s_expected = torch.tensor([[
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000]]])
    
    y_s_expected = torch.tensor([[
        [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
        [-0.5000, -0.5000, -0.5000, -0.5000, -0.5000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.5000,  0.5000,  0.5000,  0.5000,  0.5000],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000]]])
    
    torch.testing.assert_close(x_s, x_s_expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y_s, y_s_expected, atol=1e-4, rtol=1e-4)


def test_affine_grid_generator_identity(transformer):
    h, w = 5, 5

    # rotation
    degree = 45
    theta = torch.tensor([[
        [torch.cos(torch.tensor(deg2rad(degree))), -torch.sin(torch.tensor(deg2rad(degree))), 0],
        [torch.sin(torch.tensor(deg2rad(degree))), torch.cos(torch.tensor(deg2rad(degree))), 0]
    ]], dtype=torch.float32)

    batch_grids = transformer.affine_grid_generator(h, w, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    x_s_expected = torch.tensor([[
        [ 0.0000,  0.3536,  0.7071,  1.0607,  1.4142],
        [-0.3536,  0.0000,  0.3536,  0.7071,  1.0607],
        [-0.7071, -0.3536,  0.0000,  0.3536,  0.7071],
        [-1.0607, -0.7071, -0.3536,  0.0000,  0.3536],
        [-1.4142, -1.0607, -0.7071, -0.3536,  0.0000]]])
    
    y_s_expected = torch.tensor([[
        [-1.4142, -1.0607, -0.7071, -0.3536,  0.0000],
        [-1.0607, -0.7071, -0.3536,  0.0000,  0.3536],
        [-0.7071, -0.3536,  0.0000,  0.3536,  0.7071],
        [-0.3536,  0.0000,  0.3536,  0.7071,  1.0607],
        [ 0.0000,  0.3536,  0.7071,  1.0607,  1.4142]]])
    
    torch.testing.assert_close(x_s, x_s_expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y_s, y_s_expected, atol=1e-4, rtol=1e-4)


def test_bilinear_sampler(transformer):
    # load_images('data/cats/', (5, 5)) -> 1st img
    # theta: identity transform
    input = torch.tensor([[
        [[0.9804, 0.9765, 0.9765],
         [0.9412, 0.9333, 0.9216],
         [1.0000, 1.0000, 1.0000],
         [0.8980, 0.8863, 0.8745],
         [0.9765, 0.9765, 0.9725]],

        [[0.9804, 0.9725, 0.9725],
         [0.7373, 0.6980, 0.6549],
         [0.6784, 0.6510, 0.6196],
         [0.6980, 0.6588, 0.6196],
         [0.9804, 0.9725, 0.9725]],

        [[0.9922, 0.9922, 0.9882],
         [0.6784, 0.6353, 0.5804],
         [0.5569, 0.4980, 0.4275],
         [0.5804, 0.5412, 0.4941],
         [0.9765, 0.9765, 0.9725]],

        [[1.0000, 1.0000, 1.0000],
         [0.6902, 0.6588, 0.6196],
         [0.4471, 0.4118, 0.3765],
         [0.5216, 0.4980, 0.4745],
         [0.9961, 0.9961, 0.9961]],

        [[0.9843, 0.9843, 0.9843],
         [0.6314, 0.5882, 0.5373],
         [0.5059, 0.4510, 0.3765],
         [0.4863, 0.4510, 0.4078],
         [0.9490, 0.9451, 0.9451]]]])
    
    x_s = torch.tensor([[
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
        [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000]]])
    
    y_s = torch.tensor([[
        [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
        [-0.5000, -0.5000, -0.5000, -0.5000, -0.5000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.5000,  0.5000,  0.5000,  0.5000,  0.5000],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000]]])

    output = transformer.bilinear_sampler(input, x_s, y_s)

    output_expected = torch.tensor([[[
        [0.9804, 0.9765, 0.9765],
        [0.9510, 0.9441, 0.9353],
        [0.9706, 0.9667, 0.9608],
        [0.9745, 0.9716, 0.9686],
        [0.8980, 0.8863, 0.8745]],

        [[0.9804, 0.9735, 0.9735],
        [0.8363, 0.8110, 0.7846],
        [0.7735, 0.7475, 0.7181],
        [0.7561, 0.7326, 0.7069],
        [0.7480, 0.7157, 0.6833]],

        [[0.9863, 0.9824, 0.9804],
        [0.7775, 0.7456, 0.7083],
        [0.6627, 0.6206, 0.5706],
        [0.6230, 0.5809, 0.5319],
        [0.6392, 0.6000, 0.5569]],

        [[0.9941, 0.9941, 0.9912],
        [0.7596, 0.7294, 0.6904],
        [0.6054, 0.5588, 0.5025],
        [0.5385, 0.4900, 0.4333],
        [0.5657, 0.5304, 0.4892]],

        [[1.0000, 1.0000, 1.0000],
        [0.7676, 0.7441, 0.7147],
        [0.5686, 0.5353, 0.4980],
        [0.4657, 0.4333, 0.4010],
        [0.5216, 0.4980, 0.4745]]]])

    torch.testing.assert_close(output, output_expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":

    transformer = SpatialTransformer()

    test_affine_grid_generator_identity(transformer)
    test_bilinear_sampler(transformer)
    print("All tests passed.")