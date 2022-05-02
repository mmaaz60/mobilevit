import torch
from torch import nn, Tensor
from typing import Optional

from ..layers import ConvLayer, get_activation_fn, get_normalization_layer, LinearLayer, XCA
from ..modules import BaseModule
from ..misc.profiler import module_profile


class ConvDTABlock(BaseModule):
    expansion: int = 4

    def __init__(self, opts,
                 in_channels: int,
                 expan_ratio: int,
                 kernel_size: int,
                 layer_scale_init_value: Optional[float] = 1e-6,
                 dilation: Optional[int] = 1
                 ) -> None:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)

        super(ConvDTABlock, self).__init__()

        self.dwconv = ConvLayer(opts=opts, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                groups=in_channels, stride=1, use_norm=False, use_act=False, dilation=dilation)

        self.norm_xca = get_normalization_layer(opts=opts, num_features=in_channels, norm_type="layer_norm")
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(in_channels), requires_grad=True)
        self.xca = XCA(in_channels, num_heads=8, attn_dropout=0.0, bias=True)

        self.norm = get_normalization_layer(opts=opts, num_features=in_channels, norm_type="layer_norm")
        self.pwconv1 = LinearLayer(in_features=in_channels, out_features=expan_ratio * in_channels)
        self.act = get_activation_fn(act_type=act_type, inplace=inplace,
                                     negative_slope=neg_slope, num_parameters=expan_ratio * in_channels)
        self.pwconv2 = LinearLayer(in_features=expan_ratio * in_channels, out_features=in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(in_channels),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.in_channels = in_channels
        self.expan_ratio = expan_ratio
        self.kernel_size = kernel_size
        self.layer_scale_init_value = layer_scale_init_value
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        # DWConv
        out, n_params_dwconv, n_macs_dwconv = module_profile(module=self.dwconv, x=input)

        # XCA
        B, C, H, W = out.shape
        out, n_params_norm_xca, n_macs_norm_xca = module_profile(module=self.norm_xca,
                                                                 x=out.reshape(B, C, H * W).permute(0, 2, 1))
        out, n_params_xca, n_macs_xca = module_profile(module=self.xca, x=out)
        n_params_xca += n_params_norm_xca
        n_macs_xca += n_macs_norm_xca

        # PWConv
        out = out.reshape(B, H, W, C)
        out, n_params_norm, n_macs_norm = module_profile(module=self.norm, x=out)
        out, n_params_pwconv1, n_macs_pwconv1 = module_profile(module=self.pwconv1, x=out)
        n_macs_pwconv1 = n_params_pwconv1 * out.shape[0] * out.shape[1] * out.shape[2]
        out, n_params_pwconv2, n_macs_pwconv2 = module_profile(module=self.pwconv2, x=out)
        n_macs_pwconv2 = n_params_pwconv2 * out.shape[0] * out.shape[1] * out.shape[2]
        n_params_mlp = n_params_norm + n_params_pwconv1 + n_params_pwconv2
        n_macs_mlp = n_macs_norm + n_macs_pwconv1 + n_macs_pwconv2

        # Gamma (2 * -> as we have total 2 gammas)
        n_params_gamma = 2 * sum([p.numel() for p in self.gamma])
        n_macs_gamma = 2 * n_params_gamma * out.shape[0]

        return out.permute(0, 3, 1, 2), n_params_dwconv + n_params_xca + n_params_mlp + n_params_gamma, \
               n_macs_dwconv + n_macs_xca + n_macs_mlp + n_macs_gamma

    def __repr__(self) -> str:
        return '{}(in_channels={}, expan_ratio={}, kernel_size={}, layer_scale_init_value={}, dilation={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.expan_ratio,
            self.kernel_size,
            self.layer_scale_init_value,
            self.dilation
        )
