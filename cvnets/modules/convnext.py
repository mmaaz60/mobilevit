import torch
from torch import nn, Tensor
from typing import Optional

from ..layers import ConvLayer, get_activation_fn, LinearLayer
from ..modules import BaseModule
from ..misc.profiler import module_profile


class ConvNeXtBlock(BaseModule):
    expansion: int = 4

    def __init__(self, opts,
                 in_channels: int,
                 expan_ratio: int,
                 kernel_size: int,
                 layer_scale_init_value: Optional[float] = 0.0,
                 dilation: Optional[int] = 1
                 ) -> None:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)

        super(ConvNeXtBlock, self).__init__()

        self.dwconv = ConvLayer(opts=opts, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                groups=in_channels, stride=1, use_norm=True, use_act=False)
        self.pwconv1 = LinearLayer(in_features=in_channels, out_features=expan_ratio * in_channels)
        self.act = get_activation_fn(act_type=act_type, inplace=inplace,
                                     negative_slope=neg_slope, num_parameters=expan_ratio * in_channels)
        self.pwconv2 = LinearLayer(in_features=expan_ratio * in_channels, out_features=in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(in_channels),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.in_channels = in_channels
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)  # Normalization layer is the part of dwconv module
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        out, n_params_dwconv, n_macs_dwconv = module_profile(module=self.dwconv, x=input)
        out, n_params_pwconv1, n_macs_pwconv1 = module_profile(module=self.dwconv, x=out.permute(0, 2, 3, 1))
        out, n_params_pwconv2, n_macs_pwconv2 = module_profile(module=self.dwconv, x=out)

        return out.permute(0, 3, 1, 2), n_params_dwconv + n_params_pwconv1 + n_params_pwconv2, \
               n_macs_dwconv + n_macs_pwconv1 + n_macs_pwconv2

    def __repr__(self) -> str:
        return '{}(in_channels={}, expan_ratio={}, kernel_size={}, layer_scale_init_value={}, dilation={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.expan_ratio,
            self.kernel_size,
            self.layer_scale_init_value,
            self.dilation
        )
