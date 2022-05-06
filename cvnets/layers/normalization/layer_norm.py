#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor, Size
from typing import Optional, Union, List
import torch
import torch.nn.functional as F

from . import register_norm_fn


@register_norm_fn(name="layer_norm")
class LayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape: Union[int, List[int], Size],
                 eps: Optional[float] = 1e-5,
                 elementwise_affine: Optional[bool] = True
                 ):
        super(LayerNorm, self).__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


@register_norm_fn(name="layer_norm_convnext")
class LayerNormConvNext(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = sum([p.numel() for p in self.weight])
        params += sum([p.numel() for p in self.bias])
        return input, params, 0.0
