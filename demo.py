from ultralytics import YOLO

import cv2

# 加载YOLOv8预训练检测模型
model = YOLO('weights/yolov8n-pose.pt')  # 轻量版模型，速度快
#
# 运行检测
results = model('qf_img.jpg')

# 显示检测结果
results[0].show()

# 打印结果信息
print(results[0].boxes.cls.cpu().numpy())

img= results[0].plot()
cv2.imwrite("output.jpg", img)