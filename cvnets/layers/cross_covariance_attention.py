import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Optional

from .base_layer import BaseLayer
from .linear_layer import LinearLayer
from .dropout import Dropout
from ..misc.profiler import module_profile


class XCA(BaseLayer):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: Optional[float] = 0.0,
                 bias: Optional[bool] = True,
                 *args, **kwargs):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        :param attn_dropout: Attention dropout
        :param bias: Bias
        """
        super(XCA, self).__init__()
        assert embed_dim % num_heads == 0, "Got: embed_dim={} and num_heads={}".format(embed_dim, num_heads)

        self.qkv = LinearLayer(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_drop = Dropout(p=attn_dropout)
        self.proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)

        return x

    def profile_module(self, input) -> (Tensor, float, float):
        b_sz, seq_len, in_channels = input.shape
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv, x=input)
        params += p
        macs += (m * seq_len * b_sz)

        # number of operations in Q^TK
        m_qk = (seq_len * in_channels * in_channels) * b_sz
        macs += m_qk

        # number of operations in computing weighted sum
        m_wt = (seq_len * in_channels * in_channels) * b_sz
        macs += m_wt

        out_p, p, m = module_profile(module=self.proj, x=input)
        params += p
        macs += (m * seq_len * b_sz)

        return input, params, macs
