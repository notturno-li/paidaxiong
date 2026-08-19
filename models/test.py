from ultralytics import YOLO

# 加载模型
model = YOLO("your_model.pt")

# 打印类别映射字典
print(model.names)
# 输出示例: {0: 'person', 1: 'bicycle', 2: 'car', ...}
