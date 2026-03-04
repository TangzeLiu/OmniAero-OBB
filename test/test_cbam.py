import torch
import torch.nn as nn


class BipolarCrossCheckAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(BipolarCrossCheckAM, self).__init__()
        self.channels = channels

        # --- Step 1: Channel-wise Cross-Check (通道互查) ---
        self.mlp_rgb = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.mlp_ir = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.tanh = nn.Tanh()

        # --- Step 2: Spatial-wise Contour Refinement (空间轮廓) ---
        # 输入是 2 个通道 (max_pool 和 avg_pool 的堆叠)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x_rgb, x_ir):
        # --- Step 1: 通道互查逻辑 ---
        b, c, h, w = x_rgb.size()

        # 提取全局特征
        rgb_gap = self.avg_pool(x_rgb).view(b, c)
        ir_gap = self.avg_pool(x_ir).view(b, c)

        # 互查权重生成：RGB 生成给 IR 的权重，IR 生成给 RGB 的权重
        # w 的范围是 [-1, 1]
        w_ir_to_rgb = self.tanh(self.mlp_ir(ir_gap)).view(b, c, 1, 1)
        w_rgb_to_ir = self.tanh(self.mlp_rgb(rgb_gap)).view(b, c, 1, 1)

        # 修正特征：(1 + w) 的范围是 [0, 2]
        # 这一步实现了：否定(->0), 中立(->1), 增强(->2)
        x_rgb_refined = x_rgb * (1 + w_ir_to_rgb)
        x_ir_refined = x_ir * (1 + w_rgb_to_ir)

        # 均值融合：确保在双向中立(w=0)时，保持原始特征强度
        fused = (x_rgb_refined + x_ir_refined) / 2

        # --- Step 2: 空间轮廓细化 ---
        # 沿通道轴提取最大值和平均值，捕捉空间几何结构
        avg_out = torch.mean(fused, dim=1, keepdim=True)
        max_out, _ = torch.max(fused, dim=1, keepdim=True)
        spatial_info = torch.cat([avg_out, max_out], dim=1)

        # 生成空间极性权重
        w_spatial = self.tanh(self.conv_spatial(spatial_info))

        # 最终输出：对融合特征进行空间维度的奖惩
        out = fused * (1 + w_spatial)

        return out


# --- 单元测试 ---
if __name__ == "__main__":
    # 模拟输入: Batch=4, Channel=256, Size=20x20
    input_rgb = torch.randn(4, 256, 20, 20)
    input_ir = torch.randn(4, 256, 20, 20)

    model = BipolarCrossCheckAM(channels=256)
    output = model(input_rgb, input_ir)

    print(f"输入尺寸: {input_rgb.shape}")
    print(f"输出尺寸: {output.shape}")  # 尺寸保持不变，可无缝替换原 CBAM