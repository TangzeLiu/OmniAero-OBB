from ultralytics import YOLO

# 1. 加载模型结构
model = YOLO("ultralytics/cfg/models/v8/yolov8-fusion-obb.yaml")

# 2. 加载预训练权重 (迁移学习)
# YOLO 会自动跳过形状不匹配的第一层，加载后面匹配的层
try:
    model.load("yolov8n-obb.pt")
    print("预训练权重加载成功 (部分层)")
except Exception as e:
    print(f"权重加载提示: {e}")

# 3. 开始训练
model.train(
    data="dataset.yaml",  # 你的数据集配置文件
    imgsz=640,
    epochs=100,
    batch=16,  # 显存不够就改小
    workers=4,

    # === 关键参数 ===
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,  # 必须关闭 HSV，否则 4 通道会报错
    mosaic=1.0,  # 开启马赛克增强，提升小目标效果
    name="OmniAero_Fusion_test01"  # 实验名字
)