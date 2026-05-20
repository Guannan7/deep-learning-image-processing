#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习图片处理Web应用
支持批量上传图片进行目标检测或图像分类
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np

# 尝试导入深度学习库
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    import timm
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 全局模型实例
detector_model = None
classifier_model = None
classifier_transform = None
class_names = []

def load_models():
    """加载深度学习模型"""
    global detector_model, classifier_model, classifier_transform, class_names
    
    # 加载YOLOv8n检测模型
    if YOLO_AVAILABLE:
        try:
            detector_model = YOLO('yolov8n.pt')
            detector_model.conf = 0.25
            detector_model.iou = 0.45
            print("✅ YOLOv8n模型加载成功")
        except Exception as e:
            print(f"❌ YOLOv8n模型加载失败: {e}")
    
    # 加载MobileNetV4分类模型
    if CLASSIFIER_AVAILABLE:
        try:
            classifier_model = timm.create_model('mobilenetv4_conv_small', pretrained=True)
            classifier_model.eval()
            
            # 移动到可用设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            classifier_model = classifier_model.to(device)
            
            # 定义预处理
            classifier_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # 加载ImageNet类别
            import urllib.request
            import json
            url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
            with urllib.request.urlopen(url) as f:
                class_idx = json.loads(f.read().decode())
            class_names = [class_idx[str(k)][1] for k in range(1000)]
            
            print("✅ MobileNetV4模型加载成功")
        except Exception as e:
            print(f"❌ MobileNetV4模型加载失败: {e}")

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(image_path):
    """加载图片，支持jpg、png等格式，处理PNG透明通道"""
    # 尝试用OpenCV读取
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        # 尝试用PIL读取，处理一些特殊格式
        from PIL import Image
        try:
            pil_image = Image.open(image_path)
            # 如果是RGBA格式，转换为RGB
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return None, f"无法读取图片: {str(e)}"
    
    # 处理PNG透明通道（如果有alpha通道）
    if image.ndim == 4 and image.shape[2] == 4:
        # 将透明背景填充为白色
        alpha = image[:, :, 3]
        rgb = image[:, :, :3]
        white_background = np.ones_like(rgb) * 255
        alpha = alpha / 255.0
        image = (rgb * alpha[..., None] + white_background * (1 - alpha[..., None])).astype(np.uint8)
    
    # 确保是3通道图像
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return image, None

def detect_objects(image_path):
    """使用YOLOv8进行目标检测，支持jpg、png等格式"""
    if detector_model is None:
        return None, "目标检测模型未加载"
    
    try:
        # 加载图片
        image, error = load_image(image_path)
        if error is not None:
            return None, error
        
        # YOLOv8可以直接处理numpy数组（BGR格式）
        results = detector_model(image, verbose=False)[0]
        
        # 绘制检测结果
        output_image = image.copy()
        boxes = results.boxes
        
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            label = detector_model.names[class_id]
            
            # 绘制边界框
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签背景
            label_text = f"{label}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x, label_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 5
            cv2.rectangle(output_image, 
                          (label_x, label_y - label_size[1] - 5),
                          (label_x + label_size[0] + 5, label_y + 5),
                          (0, 255, 0), cv2.FILLED)
            
            # 绘制标签文字
            cv2.putText(output_image, label_text, (label_x + 2, label_y - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            detections.append({
                'label': label,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
        
        return output_image, detections
    
    except Exception as e:
        return None, f"检测失败: {str(e)}"

def classify_image(image_path):
    """使用MobileNetV4进行图像分类，支持jpg、png等格式"""
    if classifier_model is None:
        return None, "图像分类模型未加载"
    
    try:
        # 加载图片
        image, error = load_image(image_path)
        if error is not None:
            return None, error
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 预处理
        input_tensor = classifier_transform(image_rgb).unsqueeze(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        
        # 推理
        with torch.no_grad():
            outputs = classifier_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # 获取Top-5结果
        top_probs, top_indices = torch.topk(probabilities, 5)
        top_probs = top_probs.squeeze().cpu().numpy().tolist()
        top_indices = top_indices.squeeze().cpu().numpy().tolist()
        
        results = []
        for idx, prob in zip(top_indices, top_probs):
            results.append({
                'label': class_names[idx],
                'confidence': prob
            })
        
        return image, results
    
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """批量上传图片并处理"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        files = request.files.getlist('files')
        task_type = request.form.get('task_type', 'detection')  # detection 或 classification
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # 生成唯一文件名
                ext = file.filename.rsplit('.', 1)[1].lower()
                unique_name = f"{uuid.uuid4().hex}.{ext}"
                input_path = os.path.join(UPLOAD_FOLDER, unique_name)
                
                # 保存上传的文件
                file.save(input_path)
                print(f"✅ 上传文件: {file.filename} -> {input_path}")
                
                # 执行推理
                try:
                    if task_type == 'detection':
                        output_image, detections = detect_objects(input_path)
                        task_name = '目标检测'
                    else:
                        output_image, detections = classify_image(input_path)
                        task_name = '图像分类'
                    
                    # 保存输出结果
                    output_name = f"result_{unique_name}"
                    output_path = os.path.join(OUTPUT_FOLDER, output_name)
                    
                    if output_image is not None:
                        cv2.imwrite(output_path, output_image)
                        success = True
                        print(f"✅ 处理成功: {file.filename}")
                    else:
                        success = False
                        print(f"❌ 处理失败: {file.filename} - {detections}")
                    
                    results.append({
                        'original_filename': file.filename,
                        'output_filename': output_name,
                        'task_type': task_name,
                        'success': success,
                        'detections': detections if isinstance(detections, list) else []
                    })
                except Exception as e:
                    print(f"❌ 文件处理异常 {file.filename}: {str(e)}")
                    results.append({
                        'original_filename': file.filename,
                        'output_filename': '',
                        'task_type': task_type,
                        'success': False,
                        'detections': [],
                        'error': str(e)
                    })
        
        return jsonify({'results': results})
    
    except Exception as e:
        print(f"❌ API上传异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

@app.route('/outputs/<filename>')
def get_output(filename):
    """获取处理后的图片"""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/models')
def get_models():
    """获取可用模型信息"""
    return jsonify({
        'detection_available': YOLO_AVAILABLE and detector_model is not None,
        'classification_available': CLASSIFIER_AVAILABLE and classifier_model is not None
    })

if __name__ == '__main__':
    print("🚀 正在加载深度学习模型...")
    load_models()
    print("✅ 应用启动成功")
    app.run(host='0.0.0.0', port=5000, debug=True)
