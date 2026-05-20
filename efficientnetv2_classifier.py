#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNetV2-S 图像分类模型实现

模型简介:
    EfficientNetV2-S 是 Google 于2021年发布的 EfficientNetV2 系列中的小型模型。
    它通过神经架构搜索和改进的渐进学习策略，在ImageNet数据集上达到83.9%的top-1准确率，
    同时提升了训练速度和参数效率。

架构特点:
    - 混合缩放策略: 同时优化深度、宽度和分辨率
    - Fused-MBConv: 在浅层使用融合卷积，提升训练速度
    - 渐进式学习: 根据图像大小动态调整正则化强度
    - 神经架构搜索: 使用NAS优化网络结构

技术参数:
    - 参数量: ~21M
    - 计算量: 8.3 GFLOPs
    - 输入尺寸: 384x384 (最佳) / 224x224 (标准)
    - ImageNet Top-1准确率: 83.9%
    - 训练速度: 比EfficientNet-B7快5-11倍

适用场景:
    - 通用图像分类任务
    - 中等规模视觉识别系统
    - 迁移学习基础模型
    - 需要平衡精度和速度的场景

依赖包:
    - torch >= 2.0.0
    - torchvision >= 0.15.0
    - timm >= 0.9.0
    - numpy >= 1.21.0
    - opencv-python >= 4.5.0
    - matplotlib >= 3.3.0

使用示例:
    python efficientnetv2_classifier.py --image input.jpg
    python efficientnetv2_classifier.py --image input.jpg --show --topk 5 --input-size 384
"""

import argparse
import logging
import time
from typing import List, Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    import timm
except ImportError as e:
    raise ImportError(f"请安装依赖包: pip install timm -q\n错误: {e}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ImageNet 类别名称文件路径
IMAGENET_LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"


class EfficientNetV2Classifier:
    """EfficientNetV2-S 图像分类模型封装类"""

    def __init__(self, model_name: str = 'efficientnetv2_s',
                 num_classes: int = 1000, input_size: int = 384, device: str = 'cpu'):
        """
        初始化EfficientNetV2分类器

        Args:
            model_name: 模型名称，支持 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l'
            num_classes: 类别数量
            input_size: 输入图像尺寸 (224/384/480)
            device: 运行设备，可选 'cpu'、'cuda'
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.transform = None

    def _load_imagenet_labels(self) -> List[str]:
        """
        加载ImageNet类别名称

        Returns:
            List[str]: 类别名称列表
        """
        try:
            import json
            import urllib.request

            logger.info("正在加载ImageNet类别标签...")
            with urllib.request.urlopen(IMAGENET_LABELS_URL) as url:
                class_idx = json.loads(url.read().decode())
            self.class_names = [class_idx[str(k)][1] for k in range(self.num_classes)]
            logger.info(f"成功加载 {len(self.class_names)} 个类别标签")
            return self.class_names
        except Exception as e:
            logger.warning(f"加载ImageNet标签失败，使用默认标签: {e}")
            self.class_names = [f"Class_{i}" for i in range(self.num_classes)]
            return self.class_names

    def load_model(self, pretrained: bool = True) -> bool:
        """
        加载EfficientNetV2模型

        Args:
            pretrained: 是否加载预训练权重

        Returns:
            bool: 加载成功返回True，失败返回False
        """
        try:
            logger.info(f"正在加载EfficientNetV2模型: {self.model_name}")
            start_time = time.time()

            # 加载模型
            self.model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                num_classes=self.num_classes,
                img_size=self.input_size
            )

            # 移动到指定设备
            self.model = self.model.to(self.device)

            # 设置为评估模式
            self.model.eval()

            # 加载类别名称
            self._load_imagenet_labels()

            # 定义图像预处理管道
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            load_time = time.time() - start_time
            logger.info(f"模型加载成功，耗时: {load_time:.2f}秒")
            logger.info(f"运行设备: {self.device}")
            logger.info(f"输入尺寸: {self.input_size}x{self.input_size}")

            # 打印模型信息
            self._print_model_info()

            return True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False

    def _print_model_info(self):
        """打印模型信息"""
        if self.model is None:
            return

        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"模型参数总数: {total_params / 1e6:.2f}M")
        logger.info(f"可训练参数: {total_trainable_params / 1e6:.2f}M")

        # 计算FLOPs
        try:
            from thop import profile
            input_tensor = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
            flops, _ = profile(self.model, inputs=(input_tensor,))
            logger.info(f"计算量 (FLOPs): {flops / 1e9:.2f}G")
        except ImportError:
            logger.info("安装thop库可查看FLOPs: pip install thop")

    def predict(self, image: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        对单张图像进行分类预测

        Args:
            image: 输入图像，格式为BGR (OpenCV格式)
            top_k: 返回前k个预测结果

        Returns:
            Tuple[List[str], List[float]]: 类别名称列表和对应的置信度列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        try:
            start_time = time.time()

            # 图像预处理
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)

            # 获取Top-K结果
            top_probs, top_indices = torch.topk(probabilities, top_k)
            top_probs = top_probs.squeeze().cpu().numpy().tolist()
            top_indices = top_indices.squeeze().cpu().numpy().tolist()

            # 获取类别名称
            top_classes = [self.class_names[i] for i in top_indices]

            inference_time = time.time() - start_time
            logger.debug(f"分类完成，耗时: {inference_time:.2f}秒")
            logger.debug(f"Top-{top_k}预测: {list(zip(top_classes, top_probs))}")

            return top_classes, top_probs

        except Exception as e:
            logger.error(f"分类过程出错: {str(e)}")
            raise

    def draw_results(self, image: np.ndarray, classes: List[str],
                     probabilities: List[float]) -> np.ndarray:
        """
        在图像上绘制分类结果

        Args:
            image: 原始图像
            classes: 预测类别列表
            probabilities: 置信度列表

        Returns:
            np.ndarray: 绘制了结果的图像
        """
        output_image = image.copy()
        h, w = output_image.shape[:2]

        # 绘制结果背景
        start_y = 10
        line_height = 35

        for i, (cls, prob) in enumerate(zip(classes, probabilities)):
            y = start_y + i * line_height
            if y > h - 40:
                break

            # 绘制背景（半透明）
            overlay = output_image.copy()
            cv2.rectangle(overlay, (10, y - 30), (w - 10, y + 10), (0, 0, 0), cv2.FILLED)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

            # 绘制文本
            text = f"{i + 1}. {cls}: {prob:.4f}"
            cv2.putText(
                output_image,
                text,
                (15, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        return output_image

    def evaluate_performance(self, image_path: str, iterations: int = 10) -> Dict:
        """
        评估模型性能（推理时间、FPS）

        Args:
            image_path: 测试图像路径
            iterations: 测试迭代次数

        Returns:
            Dict: 性能指标字典
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 加载测试图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        logger.info(f"开始性能评估，迭代次数: {iterations}")

        # 预热运行
        for _ in range(2):
            self.predict(image, top_k=1)

        # 正式测试
        total_time = 0.0
        for i in range(iterations):
            start_time = time.time()
            self.predict(image, top_k=1)
            total_time += time.time() - start_time

        avg_time = total_time / iterations
        fps = 1.0 / avg_time

        # 获取模型信息
        total_params = sum(p.numel() for p in self.model.parameters())

        model_info = {
            'model_name': self.model_name,
            'parameters': f'{total_params / 1e6:.2f}M',
            'input_size': f'{self.input_size}x{self.input_size}',
            'num_classes': self.num_classes
        }

        logger.info(f"性能评估完成")
        logger.info(f"模型信息: {model_info}")
        logger.info(f"平均推理时间: {avg_time:.4f}秒")
        logger.info(f"帧率 (FPS): {fps:.2f}")

        return {
            'model_info': model_info,
            'average_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'iterations': iterations
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='EfficientNetV2-S 图像分类')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--output', type=str, default='output.jpg', help='输出图像路径')
    parser.add_argument('--show', action='store_true', help='是否显示分类结果')
    parser.add_argument('--topk', type=int, default=5, help='返回前K个预测结果')
    parser.add_argument('--input-size', type=int, default=384, help='输入图像尺寸 (224/384/480)')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备 (cpu/cuda)')
    parser.add_argument('--evaluate', action='store_true', help='评估模型性能')
    parser.add_argument('--iterations', type=int, default=10, help='性能评估迭代次数')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建分类器
    classifier = EfficientNetV2Classifier(
        model_name='efficientnetv2_s',
        input_size=args.input_size,
        device=args.device
    )

    # 加载模型
    if not classifier.load_model():
        logger.error("模型加载失败，程序退出")
        return

    # 性能评估模式
    if args.evaluate and args.image:
        classifier.evaluate_performance(args.image, args.iterations)
        return

    # 图像分类模式
    if args.image:
        # 加载图像
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"无法加载图像: {args.image}")
            return

        # 执行分类
        classes, probs = classifier.predict(image, top_k=args.topk)

        # 打印结果
        logger.info(f"\n分类结果 (Top-{args.topk}):")
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            logger.info(f"{i + 1}. {cls}: {prob:.4f}")

        # 绘制结果
        output_image = classifier.draw_results(image, classes, probs)

        # 保存结果
        cv2.imwrite(args.output, output_image)
        logger.info(f"分类结果已保存到: {args.output}")

        # 显示结果
        if args.show:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

    else:
        logger.info("请提供输入图像(--image)")


if __name__ == '__main__':
    main()
