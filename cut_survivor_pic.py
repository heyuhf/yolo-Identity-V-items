import os
import cv2
from datetime import datetime

# 路径配置
image_dir = 'dataset2/images/rep_zss_ss'
label_dir = 'dataset2/labels/rep_zss_ss'
output_dir = 'dataset/survivors'
os.makedirs(output_dir, exist_ok=True)

def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    """归一化坐标转为像素坐标"""
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

start = 0
count = 0
for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue

    label_path = os.path.join(label_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 筛选出指定类型 的行
    survivor_lines = [line for line in lines if line.strip().startswith('4 ')]

    if not survivor_lines:
        continue  # 如果没有求生者，跳过图片处理

    # 获取对应的图像路径
    image_name = label_file.replace('.txt', '.jpg')
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        continue

    img = cv2.imread(image_path)
    if img is None:
        continue

    img_h, img_w = img.shape[:2]

    for line in survivor_lines:
        cls_id, x, y, w, h = map(float, line.strip().split())
        x1, y1, x2, y2 = xywhn_to_xyxy(x, y, w, h, img_w, img_h)
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        today_str = datetime.now().strftime("%Y%m%d")
        out_path = os.path.join(output_dir, f'zss_{today_str}_{count+start}.jpg')
        cv2.imwrite(out_path, cropped)
        count += 1

print(f'✅ 裁剪完成，共导出 {count} 张图像。')
