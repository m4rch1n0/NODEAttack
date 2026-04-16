"""Neural ODE classifier for CIFAR-10.

Architecture from AntiNODE (ICCVW 2023), cleaned up from the original
input_attack_dopri_cifar.py. Uses torchdiffeq 0.2.5 with the canonical
NFE-counting pattern (odefunc.nfe counter).
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


def group_norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ConcatConv2d(nn.Module):
    """Conv2d that concatenates scalar time t as an extra input channel."""

    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in + 1, dim_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        return self.conv(torch.cat([tt, x], dim=1))


class ODEFunc(nn.Module):
    """ODE dynamics: GN -> ReLU -> ConcatConv -> GN -> ReLU -> ConcatConv -> GN.

    Matches the original ODEfunc from input_attack_dopri_cifar.py:107-128.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm1 = group_norm(dim)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = group_norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = group_norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):
    """Integrates ODEFunc from t=0 to t=1 using torchdiffeq.

    Resets the NFE counter on each forward pass so that model.nfe gives
    the count for the most recent call.
    """

    def __init__(self, odefunc, rtol=1e-3, atol=1e-3, method="dopri5"):
        super().__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.register_buffer("integration_time", torch.tensor([0.0, 1.0]))

    def forward(self, x):
        self.odefunc.nfe = 0
        out = odeint(
            self.odefunc, x, self.integration_time,
            rtol=self.rtol, atol=self.atol, method=self.method,
        )
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODEClassifier(nn.Module):
    """Neural ODE image classifier for CIFAR-10.

    Architecture (from AntiNODE paper):
        Conv(3->64, 3, 1) -> GN -> ReLU ->
        Conv(64->64, 4, 2, 1) -> GN -> ReLU ->
        Conv(64->64, 4, 2, 1) ->
        ODEBlock(t: 0->1, dopri5) ->
        GN -> ReLU -> AdaptiveAvgPool -> Linear(64, 10)

    Spatial dims: 32 -> 30 -> 15 -> 7 -> 7 -> 1x1 (pool)
    """

    def __init__(self, num_classes=10, dim=64,
                 rtol=1e-3, atol=1e-3, method="dopri5"):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1),
            group_norm(dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, dim, 4, 2, 1),
            group_norm(dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, dim, 4, 2, 1),
        )
        self.ode_block = ODEBlock(ODEFunc(dim), rtol=rtol, atol=atol, method=method)
        self.classifier = nn.Sequential(
            group_norm(dim),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.downsampling(x)
        x = self.ode_block(x)
        return self.classifier(x)

    @property
    def nfe(self):
        return self.ode_block.nfe

    @nfe.setter
    def nfe(self, value):
        self.ode_block.nfe = value
