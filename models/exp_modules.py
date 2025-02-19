import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.common import RepNCSPELAN4

norm_cfg_global = dict(type='SyncBN', requires_grad=True)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b),
                     stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x


class DWBlock(nn.Module):

    def __init__(self, dim, window_size, dynamic=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww

        # pw-linear
        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(dim)

        if dynamic:
            self.conv = DynamicDWConv(dim, kernel_size=window_size, stride=1,
                                      padding=window_size // 2, groups=dim)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1,
                                  padding=window_size // 2, groups=dim)

        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.GELU()

        # pw-linear
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, H, W, C = x.shape

        x = x.permute(0, 3, 1, 2)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x.permute(0, 2, 3, 1)
        return x


class DCSpatialBlock(nn.Module):
    def __init__(self, dim, window_size=7,
                 mlp_ratio=2.,  drop=0.,  drop_path=0., dynamic=True, act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.attn2conv = DWBlock(dim, window_size, dynamic)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.bn = nn.BatchNorm2d(dim)

        self.H = None
        self.W = None

    def forward(self, x):
        H, W = x.shape[2:]
        x = x.flatten(2).permute(0, 2, 1)
        B, L, C = x.shape
        # H, W = self.H, self.W
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size, remain unchanged with swin transformer
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = self.attn2conv(x)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        shortcut = x
        x = self.bn(x.view(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).view(B, H * W, C)
        x = shortcut + self.drop_path(self.mlp(x))
        return x.view(B, H, W, C).permute(0, 3, 1, 2)


class DCVit(nn.Module):
    def __init__(self, dim0: int, dim: int, n: int = 1):
        super().__init__()
        self.m = nn.Sequential(*[DCSpatialBlock(dim) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class AdaChoose(nn.Module):
    def __init__(self, dim, k1=3, k2=5):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, k1, padding=k1//2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, k2, stride=1, padding=k2//2, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class DCLG(nn.Module):
    def __init__(self, dim, mlp_ratio=4, k1=3, k2=7, drop_path=0., layer_scale_init_value=1e-2, ac=True):
        super().__init__()
        self.ada_c = AdaChoose(dim, k1, k2)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(mlp_ratio * dim, dim, 1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.ada_c(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.unsqueeze(-1).unsqueeze(-1) * x
        x = input + self.drop_path(x)
        return x


class DCLGSeq(nn.Module):
    def __init__(self, dim0: int, dim: int, n: int = 1, mlp_ratio: int = 4):
        super().__init__()
        self.m = nn.Sequential(*[DCLG(dim, mlp_ratio) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


class MHSI(nn.Module):
    def __init__(self, in_dims=[256, 128, 64], out_dim=128):
        super().__init__()
        self.lens = len(in_dims) - 1
        self.head = out_dim // 32
        self.align = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 1) for in_dim in in_dims[:-1]])
        self.shared_multi_head = nn.Conv2d(out_dim, out_dim, 1, groups=self.head)

        self.cv1 = nn.Conv2d(in_dims[-1], out_dim, 1)
        self.score = nn.Conv2d(out_dim, self.head * (len(in_dims) - 1), 1, groups=self.head)
        self.cv2 = nn.Conv2d(out_dim, out_dim, 1)
        self.cv3 = RepNCSPELAN4(out_dim * 2, out_dim, out_dim, out_dim // 2)

    def forward(self, x):
        tar = x[-1]
        tar = self.cv1(tar)
        score = self.score(tar)
        score_chunk = score.chunk(self.head, 1)
        score_chunk_sm = [F.log_softmax(x, 1).exp() for x in score_chunk]  # [self.head, 3]

        x = x[:-1]
        x = [m(i) for i, m in zip(x, self.align)]
        x = [F.interpolate(i, size=[tar.size(2), tar.size(3)],
                           mode='bilinear', align_corners=True) for i in x]
        x_chunk = [self.shared_multi_head(i).chunk(self.head, 1) for i in x]  # [3, self.head]
        out = []
        for s, x in zip(score_chunk_sm, zip(*x_chunk)):
            t = 0
            for i in range(self.lens):
                t += s[:, i].unsqueeze(1) * x[i]
            out.append(t)
        out = self.cv2(torch.cat(out, 1))
        return self.cv3(torch.cat([out, tar], 1))

