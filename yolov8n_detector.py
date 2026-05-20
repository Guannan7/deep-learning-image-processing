#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8n 目标检测模型实现

模型简介:
    YOLOv8n 是 Ultralytics 公司于2023年发布的 YOLOv8 系列中最小型的模型。
    它采用无锚框(Anchor-free)检测头设计，使用 C2f 模块作为核心构建块，
    在保持极高推理速度的同时提供良好的检测精度。

架构特点:
    - 骨干网络: CSPDarknet + C2f模块
    - 颈部网络: PAN-FPN 多尺度特征融合
    - 检测头: 解耦头设计，分别预测边界框、置信度和类别
    - Anchor-free: 直接预测目标中心点，无需预定义锚框

技术参数:
    - 参数量: ~3.2M
    - 计算量: 8.7 GFLOPs
    - 输入尺寸: 640x640
    - COCO mAP@0.5:0.95: 37.3%

适用场景:
    - 移动端部署
    - 边缘设备推理
    - 实时视频监控
    - 资源受限环境

依赖包:
    - ultralytics >= 8.0.0
    - numpy >= 1.21.0
    - opencv-python >= 4.5.0
    - matplotlib >= 3.3.0

使用示例:
    python yolov8n_detector.py --image input.jpg
    python yolov8n_detector.py --video input.mp4 --show
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError as e:
    raise ImportError(f"请安装依赖包: pip install ultralytics -q\n错误: {e}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class YOLOv8nDetector:
    """YOLOv8n 目标检测模型封装类"""

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = 'cpu'):
        """
        初始化YOLOv8n检测器

        Args:
            model_path: 模型文件路径，支持.pt权重文件或配置文件
            conf_threshold: 置信度阈值，低于此值的检测结果将被过滤
            iou_threshold: NMS的IoU阈值
            device: 运行设备，可选 'cpu'、'cuda' 或具体GPU编号
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.class_names = []

    def load_model(self) -> bool:
        """
        加载YOLOv8n模型

        Returns:
            bool: 加载成功返回True，失败返回False
        """
        try:
            logger.info(f"正在加载YOLOv8n模型: {self.model_path}")
            start_time = time.time()
            
            self.model = YOLO(self.model_path)
            
            # 设置推理参数
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            
            # 获取类别名称
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            
            load_time = time.time() - start_time
            logger.info(f"模型加载成功，耗时: {load_time:.2f}秒")
            logger.info(f"支持类别数: {len(self.class_names)}")
            logger.info(f"运行设备: {self.device}")
            
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False

    def detect(self, image: np.ndarray) -> Results:
        """
        对单张图像进行目标检测

        Args:
            image: 输入图像，格式为BGR (OpenCV格式)

        Returns:
            Results: YOLO检测结果对象，包含边界框、置信度、类别等信息
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        try:
            start_time = time.time()
            # YOLOv8默认接受BGR格式图像
            results = self.model(
                image,
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            if results:
                logger.debug(f"检测完成，耗时: {inference_time:.2f}秒")
                logger.debug(f"检测到目标数量: {len(results[0].boxes)}")
            
            return results[0] if results else None
        except Exception as e:
            logger.error(f"检测过程出错: {str(e)}")
            raise

    def draw_results(self, image: np.ndarray, results: Results,
                     show_confidence: bool = True, show_label: bool = True) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 原始图像
            results: 检测结果
            show_confidence: 是否显示置信度
            show_label: 是否显示类别标签

        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        if results is None:
            return image

        # 创建图像副本
        output_image = image.copy()
        boxes = results.boxes

        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 获取类别和置信度
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            label = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"

            # 绘制边界框
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            if show_label or show_confidence:
                label_text = ""
                if show_label:
                    label_text += label
                if show_confidence:
                    if label_text:
                        label_text += f": {confidence:.2f}"
                    else:
                        label_text += f"{confidence:.2f}"

                # 计算标签位置
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_x, label_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 5
                
                # 绘制标签背景
                cv2.rectangle(
                    output_image,
                    (label_x, label_y - label_size[1] - 5),
                    (label_x + label_size[0] + 5, label_y + 5),
                    (0, 255, 0),
                    cv2.FILLED
                )
                
                # 绘制标签文字
                cv2.putText(
                    output_image,
                    label_text,
                    (label_x + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

        return output_image

    def evaluate_performance(self, image_path: str, iterations: int = 10) -> dict:
        """
        评估模型性能（推理时间、FPS）

        Args:
            image_path: 测试图像路径
            iterations: 测试迭代次数

        Returns:
            dict: 性能指标字典，包含平均推理时间和FPS
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
            self.model(image, verbose=False)

        # 正式测试
        total_time = 0.0
        for i in range(iterations):
            start_time = time.time()
            self.model(image, verbose=False)
            total_time += time.time() - start_time

        avg_time = total_time / iterations
        fps = 1.0 / avg_time

        logger.info(f"性能评估完成")
        logger.info(f"平均推理时间: {avg_time:.4f}秒")
        logger.info(f"帧率 (FPS): {fps:.2f}")

        return {
            'average_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'iterations': iterations
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8n 目标检测')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--video', type=str, help='输入视频路径')
    parser.add_argument('--output', type=str, default='output.jpg', help='输出图像/视频路径')
    parser.add_argument('--show', action='store_true', help='是否显示检测结果')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU阈值')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备 (cpu/cuda)')
    parser.add_argument('--evaluate', action='store_true', help='评估模型性能')
    parser.add_argument('--iterations', type=int, default=10, help='性能评估迭代次数')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建检测器
    detector = YOLOv8nDetector(
        model_path='yolov8n.pt',
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )

    # 加载模型
    if not detector.load_model():
        logger.error("模型加载失败，程序退出")
        return

    # 性能评估模式
    if args.evaluate and args.image:
        detector.evaluate_performance(args.image, args.iterations)
        return

    # 图像检测模式
    if args.image:
        # 加载图像
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"无法加载图像: {args.image}")
            return

        # 执行检测
        results = detector.detect(image)

        # 绘制结果
        output_image = detector.draw_results(image, results)

        # 保存结果
        cv2.imwrite(args.output, output_image)
        logger.info(f"检测结果已保存到: {args.output}")

        # 显示结果
        if args.show:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

    # 视频检测模式
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {args.video}")
            return

        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        logger.info(f"开始处理视频: {args.video}")
        frame_count = 0
        total_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = detector.detect(frame)
            output_frame = detector.draw_results(frame, results)
            total_time += time.time() - start_time

            out.write(output_frame)
            frame_count += 1

            if args.show:
                cv2.imshow('YOLOv8n Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        avg_fps = frame_count / total_time if total_time > 0 else 0
        logger.info(f"视频处理完成，共 {frame_count} 帧")
        logger.info(f"平均处理帧率: {avg_fps:.2f} FPS")
        logger.info(f"结果已保存到: {args.output}")

    else:
        logger.info("请提供输入图像(--image)或视频(--video)")


if __name__ == '__main__':
    main()
