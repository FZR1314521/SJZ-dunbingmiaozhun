"""
CNN训练脚本 - 准心状态分类器

功能：
1. 加载已瞄准状态和未瞄准状态的数据
2. 构建简单的CNN模型
3. 训练模型并保存

使用方法：
python train_cnn.py
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

def load_images_from_dir(directory, label):
    """
    从目录加载图片并标注标签
    
    参数：
        directory: 图片目录路径
        label: 标签（0或1）
        
    返回：
        (images, labels) 元组
    """
    images = []
    labels = []
    
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(directory, filename)
            try:
                # 使用imdecode处理中文路径
                img_array = np.fromfile(filepath, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # 调整大小到64x64
                    img = cv2.resize(img, (64, 64))
                    # BGR转RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"读取图片失败: {filepath}, 错误: {e}")
    
    return images, labels

def build_cnn_model(input_shape=(64, 64, 3), num_classes=2):
    """
    构建简单的CNN模型
    
    参数：
        input_shape: 输入形状
        num_classes: 类别数
        
    返回：
        编译好的模型
    """
    model = models.Sequential([
        # 第一层卷积
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # 第二层卷积
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 第三层卷积
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 展平
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # 数据目录
    aimed_dir = "准心数据集/已瞄准状态"
    not_aimed_dir = "准心数据集/未瞄准状态"
    
    print("加载数据...")
    
    # 加载已瞄准状态数据（标签1）
    aimed_images, aimed_labels = load_images_from_dir(aimed_dir, 1)
    print(f"已瞄准状态: {len(aimed_images)} 张图片")
    
    # 加载未瞄准状态数据（标签0）
    not_aimed_images, not_aimed_labels = load_images_from_dir(not_aimed_dir, 0)
    print(f"未瞄准状态: {len(not_aimed_images)} 张图片")
    
    # 合并数据
    X = np.array(aimed_images + not_aimed_images, dtype=np.float32) / 255.0
    y = np.array(aimed_labels + not_aimed_labels, dtype=np.int32)
    
    print(f"\n总数据量: {X.shape[0]} 张")
    print(f"图片形状: {X.shape[1:]}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {X_train.shape[0]} 张")
    print(f"测试集: {X_test.shape[0]} 张")
    
    # 构建模型
    print("\n构建CNN模型...")
    model = build_cnn_model()
    model.summary()
    
    # 训练模型
    print("\n开始训练...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # 评估模型
    print("\n评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试准确率: {test_acc:.4f}")
    
    # 预测
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['未瞄准', '已瞄准']))
    
    # 保存模型
    model.save("准心状态分类器.h5")
    print("\n模型已保存为: 准心状态分类器.h5")
    
    # 保存模型权重
    model.save_weights("准心状态分类器_weights.h5")
    print("模型权重已保存为: 准心状态分类器_weights.h5")

if __name__ == "__main__":
    main()
