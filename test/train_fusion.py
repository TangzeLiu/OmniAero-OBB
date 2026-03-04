from ultralytics import YOLO


if __name__ == '__main__':
    # 1. 加载模型结构
    model = YOLO("ultralytics/cfg/models/v8/yolov8-cbam.yaml")

    # 2. 加载预训练权重 (迁移学习)
    # YOLO 会自动跳过形状不匹配的第一层，加载后面匹配的层
    try:
        model.load("yolov8n-obb.pt")
        print("预训练权重加载成功 (部分层)")
    except Exception as e:
        print(f"权重加载提示: {e}")

    # 3. 开始训练
    model.train(
        data="F:/work/OmniAero-OBB/test/dataset.yaml",
        # F:\work\OmniAero-OBB\test\dataset.yaml
        # /mnt/workspace/OmniAero-OBB/test/dataset.yaml
        imgsz=800,           # 【提升】从 640 提升到 800，增强小目标识别
        epochs=150,          # 【增加】给大数据集更多学习时间
        batch=56,            # 【提升】20G 显存建议从 64 起步试试
        workers=8,          # 【提升】加快数据加载
        device=0,
        amp=True,
        patience=50,         # 【新增】50次迭代没提升再停止
        # === 关键参数 ===
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, 
        mosaic=1.0, 
        mixup=0.1,           # 【新增】增加 mixup 增强，防止过拟合
        name="OmniAero_Fusion_HighRes" 
    )