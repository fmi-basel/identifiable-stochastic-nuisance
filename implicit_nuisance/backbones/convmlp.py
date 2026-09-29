import torch
import torch.nn as nn
import numpy as np


def Conv2dMLP(input_dim, output_dim, n_in_channels=1, channel_dim=512, n_layers=1, 
              kernel_size=[(5,5)], stride=[(2,2)], padding=[(2,2)], max_pool_size=[(2,2)], 
              use_batch_norm=False):

    backbone = nn.Sequential()

    # if kernel_size, stride, padding, and max_pool_size are only given for one layer, repeat them for n_layers
    if len(kernel_size) == 1 and n_layers > 1:
        kernel_size = kernel_size * n_layers
        stride = stride * n_layers
        if padding is not None:
            padding = padding * n_layers
        if max_pool_size is not None:
            max_pool_size = max_pool_size * n_layers

    for i in range(n_layers):
        in_channels = n_in_channels if i == 0 else channel_dim
        
        backbone.add_module(
            f"conv{i}",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channel_dim,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i] if padding is not None else 0
            )
        )
        
        if use_batch_norm:
            backbone.add_module(f"batchNorm{i}", nn.BatchNorm2d(channel_dim))
        backbone.add_module(f"relu{i}", nn.ReLU())
        if max_pool_size is not None:
            backbone.add_module(f"pool{i}", nn.MaxPool2d(max_pool_size[i]))

    
    kernel_size = np.array(kernel_size)[:, 0]
    stride = np.array(stride)[:, 0]
    padding = np.array(padding)[:, 0] if padding is not None else np.zeros(n_layers)
    max_pool_size = np.array(max_pool_size)[:, 0] if max_pool_size is not None else np.ones(n_layers)
    channel_dim = channel_dim

    backbone.add_module(
        "flatten",
        nn.Flatten()
    )
    backbone.add_module(
        "linear",
        nn.Linear(
            in_features=get_output_spatial_shape(input_dim, kernel_size, stride, padding, max_pool_size) * channel_dim,
            out_features=output_dim
        )
    )

    return backbone

def get_output_spatial_shape(input_shape, kernel_size, stride, padding, max_pool_size=None):
    shape = input_shape if isinstance(input_shape, int) else input_shape[0]
    for i in range(len(kernel_size)):
        shape = (shape - kernel_size[i] + 2 * padding[i]) // stride[i] + 1
        if max_pool_size is not None:
            shape = (shape - max_pool_size[i]) // max_pool_size[i] + 1
    return int(shape)