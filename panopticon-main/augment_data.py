"""
数据增强脚本 - 用于扩充准心训练数据集

功能：
1. 读取已瞄准状态的图片
2. 应用多种数据增强技术：水平翻转、垂直翻转、旋转、亮度调整等
3. 保存增强后的图片到原目录

使用方法：
python augment_data.py
"""

import os
import cv2
import numpy as np
from pathlib import Path

# 设置OpenCV使用中文路径
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'

def augment_image(image):
    """
    对单张图片进行多种数据增强
    
    参数：
        image: BGR格式的numpy数组
        
    返回：
        增强后的图片列表
    """
    augmented_images = []
    
    # 原始图片（也保存一份，便于统计）
    augmented_images.append(("original", image.copy()))
    
    # 水平翻转
    flipped_h = cv2.flip(image, 1)
    augmented_images.append(("flip_h", flipped_h))
    
    # 垂直翻转
    flipped_v = cv2.flip(image, 0)
    augmented_images.append(("flip_v", flipped_v))
    
    # 水平+垂直翻转
    flipped_hv = cv2.flip(image, -1)
    augmented_images.append(("flip_hv", flipped_hv))
    
    # 旋转90度
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(("rot_90", rotated_90))
    
    # 旋转180度
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    augmented_images.append(("rot_180", rotated_180))
    
    # 旋转270度
    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    augmented_images.append(("rot_270", rotated_270))
    
    # 轻微亮度调整（+10%）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(("bright", brightened))
    
    # 轻微变暗（-10%）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.9, 0, 255)
    darkened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(("dark", darkened))
    
    # 轻微对比度调整（+10%）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.1, 0, 255)
    contrast_up = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    augmented_images.append(("contrast_up", contrast_up))
    
    # 轻微模糊
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    augmented_images.append(("blur", blurred))
    
    return augmented_images

def main():
    # 已瞄准状态数据目录
    input_dir = Path("准心数据集/已瞄准状态")
    
    # 获取所有图片文件
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = []
    
    # 使用glob来处理中文路径
    for ext in image_extensions:
        pattern = str(input_dir / f"*{ext}")
        import glob
        image_files.extend(glob.glob(pattern))
    
    # 去重并排序
    image_files = sorted(set(image_files))
    
    if not image_files:
        print("未找到任何图片文件")
        return
    
    print(f"找到 {len(image_files)} 张原始图片")
    print("开始数据增强...")
    
    total_augmented = 0
    
    for i, image_path in enumerate(image_files):
        # 使用imdecode处理中文路径
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue
        
        # 进行数据增强
        augmented = augment_image(image)
        
        # 保存增强后的图片
        # 提取文件名和扩展名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        
        for augment_type, aug_image in augmented:
            # 生成新文件名：原文件名_增强类型_序号.jpg
            output_name = f"{base_name}_{augment_type}{ext}"
            output_path = str(input_dir / output_name)
            
            # 使用imencode处理中文路径保存
            result, encoded = cv2.imencode(ext, aug_image)
            if result:
                with open(output_path, 'wb') as f:
                    encoded.tofile(f)
                total_augmented += 1
        
        # 进度显示
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(image_files)} 张图片")
    
    print(f"\n数据增强完成！")
    print(f"原始图片: {len(image_files)} 张")
    print(f"增强后图片: {total_augmented} 张")
    print(f"保存在: {input_dir.absolute()}")

if __name__ == "__main__":
    main()
