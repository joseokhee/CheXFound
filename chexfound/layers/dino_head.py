# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class _WeightNormLinear(nn.Module):
    """Drop-in replacement for weight_norm(nn.Linear) that works with any dtype.
    Equivalent to weight_norm with weight_g=1: output = x @ normalize(weight, dim=1)^T.
    Avoids 'weight_norm_fwd_first_dim_kernel' which is not implemented for bf16.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_v = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_g = nn.Parameter(torch.ones(out_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight_v, dim=1) * self.weight_g
        return F.linear(x, weight)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = _WeightNormLinear(bottleneck_dim, out_dim)
        trunc_normal_(self.last_layer.weight_v, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
