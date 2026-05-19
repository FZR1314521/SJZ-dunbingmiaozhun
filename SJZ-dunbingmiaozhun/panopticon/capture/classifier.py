"""
准心状态分类器 - 用于实时判断准心状态

功能：
1. 加载训练好的CNN模型
2. 对输入图片进行推理
3. 返回瞄准/未瞄准状态
"""

import os
import cv2
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras import layers, models
except ImportError:
    tf = None
    load_model = None
    layers = None
    models = None

class AimingClassifier:
    """
    准心状态分类器
    
    加载训练好的CNN模型，用于实时判断准心是瞄准状态还是未瞄准状态。
    """
    
    def __init__(self, model_path="准心状态分类器.h5"):
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _build_default_model(self):
        """构建默认模型架构（用于加载权重）"""
        if models is None or layers is None:
            return None
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])
        
        return model
    
    def _load_model(self):
        """加载训练好的模型"""
        if load_model is None:
            print("警告：TensorFlow未安装，无法加载模型")
            return
        
        # 首先尝试加载完整模型
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"模型加载成功: {self.model_path}")
                return
            except Exception as e:
                print(f"直接加载模型失败: {e}")
                print("尝试加载权重文件...")
        
        # 尝试加载权重文件
        weights_path = "准心状态分类器_weights.h5"
        if os.path.exists(weights_path):
            try:
                self.model = self._build_default_model()
                if self.model:
                    self.model.load_weights(weights_path)
                    print(f"权重加载成功: {weights_path}")
            except Exception as e:
                print(f"加载权重失败: {e}")
                self.model = None
        else:
            print(f"警告：权重文件不存在: {weights_path}")
    
    def predict(self, image):
        """
        预测准心状态
        
        参数：
            image: BGR格式的numpy数组
            
        返回：
            dict: {'status': 'aimed'|'not_aimed', 'confidence': 置信度}
        """
        if self.model is None:
            return {'status': 'unknown', 'confidence': 0.0}
        
        try:
            # 调整大小到64x64
            img = cv2.resize(image, (64, 64))
            # BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 归一化
            img = img / 255.0
            # 添加batch维度
            img = np.expand_dims(img, axis=0)
            
            # 预测
            pred = self.model.predict(img, verbose=0)
            class_idx = np.argmax(pred[0])
            confidence = float(pred[0][class_idx])
            
            # 0: 未瞄准, 1: 已瞄准
            status = 'aimed' if class_idx == 1 else 'not_aimed'
            
            return {'status': status, 'confidence': confidence}
        
        except Exception as e:
            print(f"预测失败: {e}")
            return {'status': 'unknown', 'confidence': 0.0}
    
    @property
    def is_ready(self):
        """模型是否就绪"""
        return self.model is not None
