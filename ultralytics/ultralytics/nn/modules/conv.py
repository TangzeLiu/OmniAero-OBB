# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "Fusion",
    "SCBFusion",
    "CBAMFusion"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1,c2=None, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


# Zhang written
class Fusion(nn.Module):
    def __init__(self,c1,c2,k=3,s=2,p=None,g=1,d=1,act=True):
        """Initialize Fusion module.
        Args:
            c1 输入通道数
            c2 融合后的输出通道数
            k   卷积核大小
            s   卷积核步长
            p   填充
            g   分组
            d   膨胀率
            act
            """
        super().__init__()
        # from .conv import autopad
        pad = autopad(k,p,d)    # autopad 会返回一个值或一个列表，代表补几圈0

        self.conv_rgb =  nn.Conv2d(3,c2,k,s,pad,groups=g,dilation=d,bias=False)
        self.conv_ir  =  nn.Conv2d(1,c2,k,s,pad,groups=g,dilation=d,bias=False)

        hidden_dim = max(8, c2 // 2)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 第一步：全局平均池化 (压缩空间维度)
            nn.Conv2d(c2 * 2, hidden_dim, 1),  # 第二步：降维挤压 (Squeeze)
            nn.ReLU(inplace=True),  # 第三步：非线性激活 (赋予逻辑思考能力)
            nn.Conv2d(hidden_dim, 2, 1),  # 第四步：升维输出 (Excite -> 输出 2 个权重)
            nn.Softmax(dim=1)  # 第五步：互补归一化 (W_rgb + W_ir = 1)
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):

        if x.shape[1] == 3:
            # 如果输入是 3 通道（说明是 YOLO 初始化在测 stride）
            x_rgb = x  # RGB 直接拿这 3 个通道
            x_ir = torch.zeros_like(x[:, 0:1, :, :])  # 伪造 1 个同尺寸的全 0 通道给 IR
        else:
            # 如果输入 >= 4 通道（说明是真正的训练数据进来了）
            x_rgb = x[:, :3, :, :]  # 提取前 3 个通道 (RGB)
            x_ir = x[:, 3:4, :, :]  # 提取第 4 个通道 (IR)

        # --- 步骤 B: 独立空间特征提取 (Spatial Feature Extraction) ---
        feat_rgb = self.conv_rgb(x_rgb)  # 得到 RGB 高级特征 [B, c2, H, W]
        feat_ir = self.conv_ir(x_ir)  # 得到 IR 高级特征 [B, c2, H, W]

        # --- 步骤 C: 跨模态不确定性推理 (Cross-Modal Uncertainty Inference) ---
        # 拼接特征，让网络同时“看”到两个模态的好坏
        cat_feat = torch.cat([feat_rgb, feat_ir], dim=1)  # [B, 2*c2, H, W]

        # 计算动态权重 [B, 2, 1, 1]
        weights = self.attention(cat_feat)

        # 拆分权重 (此时 w_rgb 和 w_ir 是两个加起来等于 1 的动态浮点数)
        w_rgb = weights[:, 0:1, :, :]
        w_ir = weights[:, 1:2, :, :]

        # --- 步骤 D: 互补加权融合 (Complementary Weighted Fusion) ---
        # 广播机制将权重应用到整张特征图的每一个像素上
        feat_fused = (feat_rgb * w_rgb) + (feat_ir * w_ir)

        # --- 步骤 E: 输出 ---
        return self.act(self.bn(feat_fused))

# ====================================================================
# 模块名：SCBFusion (Spatial Cross-Bipolar Fusion)
# 作用：作为Stem层，负责多模态数据的解耦、极性评估、排雷与融合
# ====================================================================
class SCBFusion(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        """
        c1: 输入通道数 (4)
        c2: 输出通道数 (如 64)
        """
        super().__init__()
        # 使用 YOLO 官方自适应填充
        from .conv import autopad
        pad = autopad(k, p, d)

        # 1. 模态物理拆分与独立特征提取
        # RGB支路 (3通道)
        self.conv_rgb = nn.Conv2d(3, c2, k, s, pad, groups=g, dilation=d, bias=False)
        # IR支路 (1通道)
        self.conv_ir = nn.Conv2d(1, c2, k, s, pad, groups=g, dilation=d, bias=False)

        # 2. 联合极性评估头 (Bipolar Assessment Head)
        # 作用：生成 [-1, 1] 的空间极性图
        hidden_dim = max(16, c2 // 4)
        self.polarity_head = nn.Sequential(
            nn.Conv2d(c2 * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # 输出 2 个通道，分别对应 M_rgb 和 M_ir
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
            # 核心：Tanh 映射到 [-1, 1]
            nn.Tanh()
        )

        # 3. 后处理
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # --- 兼容性修复：处理 YOLO 初始化时的 3 通道测试张量 ---
        if x.shape[1] == 3:
            # 伪造 IR 数据，防止代码崩溃
            x_rgb = x
            x_ir = torch.zeros_like(x[:, 0:1, :, :])
        else:
            # 正常的 4 通道数据 (B, 4, H, W)
            x_rgb = x[:, :3, :, :]
            x_ir = x[:, 3:4, :, :]

        # A. 提取独立特征
        f_rgb = self.conv_rgb(x_rgb)
        f_ir = self.conv_ir(x_ir)

        # B. 生成空间极性图
        # Concat 让网络拥有全局对比视野
        polarized_maps = self.polarity_head(torch.cat([f_rgb, f_ir], dim=1))
        M_rgb = polarized_maps[:, 0:1, :, :]  # RGB 的自信度 [-1, 1]
        M_ir = polarized_maps[:, 1:2, :, :]  # IR 的自信度 [-1, 1]

        # ========================================================
        # C. 核心逻辑：交叉一票否决 (Cross-Veto Mechanism)
        # ========================================================

        # 1. 自身增益 (Boost): 越自信，特征越强 (0~2)
        boost_rgb = 1.0 + M_rgb
        boost_ir = 1.0 + M_ir

        # 2. 交叉许可 (Gate): 只有对方不反对(-1)，我才能通过
        # Clamp(..., 0, 1) 确保 Gate 不会变成负数
        gate_rgb_to_ir = torch.clamp(1.0 + M_rgb, min=0.0, max=1.0)  # RGB 给 IR 的通行证
        gate_ir_to_rgb = torch.clamp(1.0 + M_ir, min=0.0, max=1.0)  # IR 给 RGB 的通行证

        # 3. 最终权重计算
        w_rgb = boost_rgb * gate_ir_to_rgb
        w_ir = boost_ir * gate_rgb_to_ir

        # D. 协同融合
        fused_feat = (f_rgb * w_rgb) + (f_ir * w_ir)

        # E. 输出
        return self.act(self.bn(fused_feat))


class CBAMFusion(nn.Module):
    def __init__(self,c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()

        # --- 第一阶段：原位审计 (4通道对撞) ---
        # 专门针对原始 R,G,B (3ch) 和 T (1ch) 的互查
        self.mlp_rgb = nn.Sequential(nn.Linear(3, 1), nn.ReLU(), nn.Linear(1, 1))
        self.mlp_t = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))

        # --- 第二阶段：特征提取 ---
        # 语义投影：将审计后的 4 通道映射到高维 c2
        self.conv_semantic = nn.Conv2d(4, c2, kernel_size=1, stride=s, bias=False)
        # 几何投影：3x3 提取轮廓
        self.conv_contour = nn.Conv2d(4, 1, kernel_size=3, stride=s, padding=1, bias=False)

        # --- 第三阶段：空间细化 ---
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

        self.tanh = nn.Tanh()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else nn.Identity()

    def forward(self, x):


        # x: [B, 4, H, W]
        b, c_in, h, w = x.size()

        if c_in == 3:
            # 如果输入只有 3 通道，我们补一个全 0 的第 4 通道 (T)
            x_rgb = x
            x_t = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
        else:
            # 正常训练/推理时的 4 通道逻辑
            x_rgb = x[:, :3, :, :]
            x_t = x[:, 3:4, :, :]


        # --- Step 1: 原位互查 (控制核心) ---
        # 提取原始 4 通道的全局统计量
        raw_rgb_gap = self.avg_pool(x_rgb).view(b, 3)
        raw_t_gap = self.avg_pool(x_t).view(b, 1)

        # 产生 -1 到 1 的权重
        w_t_to_rgb = self.tanh(self.mlp_t(raw_t_gap)).view(b, 1, 1, 1)
        w_rgb_to_t = self.tanh(self.mlp_rgb(raw_rgb_gap)).view(b, 1, 1, 1)

        # 修正原始 4 通道
        x_refined = torch.cat([
            x_rgb * (1 + w_t_to_rgb),
            x_t * (1 + w_rgb_to_t)
        ], dim=1)

        # --- Step 2: 分支投影 ---
        # 此时 conv_semantic 处理的是已经“洗干净”的颜色数据
        feat_color = self.conv_semantic(x_refined)
        # 此时 conv_contour 提取的是逻辑一致的轮廓
        feat_contour = self.conv_contour(x_refined)

        # --- Step 3: 空间锚定 ---
        semantic_map, _ = torch.max(feat_color, dim=1, keepdim=True)
        spatial_info = torch.cat([semantic_map, feat_contour], dim=1)
        w_spatial = self.tanh(self.conv_spatial(spatial_info))

        out = feat_color * (1 + w_spatial)
        return self.act(self.bn(out))