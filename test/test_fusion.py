import torch
import torch.nn as nn


# ================= 这里放你修复后的 Fusion 类 =================
def autopad(k, p=None, d=1):  # 模拟 yolov8 的 autopad
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Fusion(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()
        pad = autopad(k, p, d)

        # 修复后的卷积通道
        self.conv_rgb = nn.Conv2d(3, c2, k, s, pad, groups=g, dilation=d, bias=False)
        self.conv_ir = nn.Conv2d(1, c2, k, s, pad, groups=g, dilation=d, bias=False)

        hidden_dim = max(8, c2 // 2)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2 * 2, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x_rgb, x_ir = x[:, :3, :, :], x[:, 3:4, :, :]
        feat_rgb, feat_ir = self.conv_rgb(x_rgb), self.conv_ir(x_ir)

        cat_feat = torch.cat([feat_rgb, feat_ir], dim=1)
        weights = self.attention(cat_feat)

        # 为了测试，我们把权重打印出来看看
        # print(f"RGB 权重: {weights[0, 0, 0, 0]:.4f}, IR 权重: {weights[0, 1, 0, 0]:.4f}")

        w_rgb, w_ir = weights[:, 0:1, :, :], weights[:, 1:2, :, :]
        feat_fused = (feat_rgb * w_rgb) + (feat_ir * w_ir)
        return self.act(self.bn(feat_fused))


# ==========================================================

if __name__ == "__main__":
    print("开始测试 Fusion 模块...")

    # 1. 实例化模型
    # 假设输入 4 通道，输出 64 通道，卷积核为 3，步长为 2 (下采样)
    model = Fusion(c1=4, c2=64, k=3, s=2)

    # 2. 伪造假数据：模拟 Batch_size=2, 4通道(RGB+IR), 图像大小 640x640
    dummy_input = torch.randn(2, 4, 640, 640)
    print(f"输入张量维度: {dummy_input.shape}")

    # 3. 前向传播 (模拟送入网络)
    output = model(dummy_input)

    # 4. 验证输出
    print(f"输出张量维度: {output.shape}")

    # 检查期望结果
    assert output.shape == (2, 64, 320, 320), "测试失败！输出维度不对！"
    print("✅ 测试完美通过！模块可以正常工作并进行了尺寸缩放 (640 -> 320)。")