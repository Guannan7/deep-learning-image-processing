# 🚀 深度学习模型新手小白使用手册

欢迎来到深度学习模型的世界！这份手册专为零基础的小伙伴准备，带你轻松上手目标检测和图像分类任务。

---

## 📋 目录

1. [环境检查](#1-环境检查)
2. [安装依赖](#2-安装依赖)
3. [Web应用快速上手](#3-web应用快速上手)
4. [目标检测模型使用](#4-目标检测模型使用)
   - [YOLOv8n 超轻量检测](#41-yolov8n-超轻量检测)
   - [YOLOv8s 平衡检测](#42-yolov8s-平衡检测)
5. [图像分类模型使用](#5-图像分类模型使用)
   - [MobileNetV4-Conv-S 移动端分类](#51-mobilenetv4-conv-s-移动端分类)
   - [EfficientNetV2-S 高精度分类](#52-efficientnetv2-s-高精度分类)
6. [常见问题](#6-常见问题)
7. [示例输出解读](#7-示例输出解读)

---

## 1️⃣ 环境检查

首先检查你的电脑是否已经安装了 Python 和必要的依赖。

### 检查 Python 版本
打开命令提示符（Windows）或终端（Mac/Linux），输入：
```bash
python --version
```
✅ 建议版本：Python 3.8 及以上

### 检查已安装的依赖
```bash
pip list | findstr "torch ultralytics timm flask"
```
如果显示版本号，说明已安装；如果没有显示，需要安装。

---

## 2️⃣ 安装依赖

### 方式一：一键安装（推荐）
打开命令提示符，复制粘贴下面的命令：

```bash
# 安装所有依赖（YOLOv8 + 分类模型 + Web应用）
pip install ultralytics torch torchvision timm opencv-python matplotlib numpy flask flask-cors pillow -q
```

### 方式二：分开安装
如果只想安装部分模型的依赖：

```bash
# 仅安装 YOLOv8 目标检测依赖
pip install ultralytics opencv-python matplotlib numpy -q

# 仅安装图像分类依赖
pip install torch torchvision timm opencv-python matplotlib numpy -q

# 仅安装Web应用依赖
pip install flask flask-cors pillow -q
```

### 验证安装是否成功
```bash
pip list | findstr "ultralytics torch timm opencv-python flask"
```
如果显示类似下面的内容，说明安装成功：
```
ultralytics               8.4.51
torch                     2.12.0
timm                      1.0.27
opencv-python             4.13.0.92
flask                     2.3.3
```

---

## 3️⃣ Web应用快速上手

**推荐使用Web界面！** 简单直观，无需命令行操作。

### 启动Web应用
```bash
cd e:\Project\deep learning
python app.py
```

### 访问地址
- **本地访问**: http://127.0.0.1:5000
- **局域网访问**: http://你的IP地址:5000

### Web应用功能
| 功能 | 说明 |
|------|------|
| 📤 批量上传 | 支持点击上传和拖拽上传，一次可上传多张图片 |
| 🎯 目标检测 | 识别图片中的物体并画出边界框 |
| 🏷️ 图像分类 | 判断图片中主要物体是什么 |
| 🔗 分组展示 | 自动将相似类别的图片分组显示 |
| ⬇️ 一键下载 | 批量下载处理后的图片 |

### 使用步骤
1. 打开浏览器访问 http://127.0.0.1:5000
2. 选择任务类型：目标检测 或 图像分类
3. 点击或拖拽图片到上传区域
4. 点击「开始处理」按钮
5. （可选）勾选「按类别分组」查看分组结果
6. 点击「下载全部结果」保存图片

---

## 4️⃣ 目标检测模型使用

目标检测可以识别图像中的物体，并画出框框标出位置。

### 4.1 YOLOv8n 超轻量检测

**适合场景**：手机、树莓派等资源有限的设备

#### 使用步骤：

1. **准备一张测试图片**（比如 `test.jpg`），放在代码文件同一文件夹

2. **打开命令提示符**，进入代码所在文件夹：
```bash
cd e:\Project\deep learning
```

3. **运行检测命令**：
```bash
python yolov8n_detector.py --image test.jpg --show
```

#### 常用参数说明：
| 参数 | 作用 | 示例 |
|------|------|------|
| `--image` | 指定输入图片 | `--image test.jpg` |
| `--video` | 指定输入视频 | `--video test.mp4` |
| `--output` | 指定输出文件 | `--output result.jpg` |
| `--show` | 显示检测结果 | `--show` |
| `--conf` | 置信度阈值（0-1） | `--conf 0.3` |
| `--device` | 使用CPU或GPU | `--device cuda` |

#### 完整命令示例：
```bash
# 检测图片并显示结果
python yolov8n_detector.py --image test.jpg --show

# 检测视频并保存结果
python yolov8n_detector.py --video input.mp4 --output output.mp4

# 使用GPU加速（需要NVIDIA显卡）
python yolov8n_detector.py --image test.jpg --device cuda
```

### 4.2 YOLOv8s 平衡检测

**适合场景**：需要更高精度的桌面端应用

#### 使用步骤：
```bash
# 基本使用
python yolov8s_detector.py --image test.jpg --show

# 评估模型性能
python yolov8s_detector.py --image test.jpg --evaluate --iterations 10
```

#### YOLOv8n vs YOLOv8s 对比：
| 特性 | YOLOv8n | YOLOv8s |
|------|---------|---------|
| 模型大小 | 小 | 中等 |
| 检测速度 | 快 | 中等 |
| 检测精度 | 中等 | 高 |
| 适合设备 | 移动端 | 桌面端 |

---

## 5️⃣ 图像分类模型使用

图像分类可以识别图片中主要物体是什么（比如猫、狗、汽车等）。

### 5.1 MobileNetV4-Conv-S 移动端分类

**适合场景**：手机APP、嵌入式设备

#### 使用步骤：
```bash
# 基本分类
python mobilenetv4_classifier.py --image test.jpg --show

# 查看前3个预测结果
python mobilenetv4_classifier.py --image test.jpg --topk 3
```

### 5.2 EfficientNetV2-S 高精度分类

**适合场景**：需要高准确率的分类任务

#### 使用步骤：
```bash
# 基本分类（默认输入尺寸224）
python efficientnetv2_classifier.py --image test.jpg --show

# 使用更高分辨率（推荐，精度更高）
python efficientnetv2_classifier.py --image test.jpg --input-size 384 --show

# 使用GPU加速
python efficientnetv2_classifier.py --image test.jpg --input-size 384 --device cuda
```

#### 常用参数说明：
| 参数 | 作用 | 示例 |
|------|------|------|
| `--image` | 指定输入图片 | `--image test.jpg` |
| `--topk` | 返回前K个预测结果 | `--topk 5` |
| `--input-size` | 输入图像尺寸 | `--input-size 384` |
| `--device` | 使用CPU或GPU | `--device cuda` |
| `--evaluate` | 评估性能 | `--evaluate` |

---

## 6️⃣ 常见问题

### Q1：运行时提示 "找不到模块"
**错误信息**：`ModuleNotFoundError: No module named 'ultralytics'`

**解决方法**：
```bash
pip install ultralytics -q
```

### Q2：运行时提示 "找不到图片"
**错误信息**：`无法加载图像: test.jpg`

**解决方法**：
1. 确保图片文件存在于代码同一文件夹
2. 确保文件名拼写正确（注意大小写）
3. 使用完整路径：`--image "C:\Users\xxx\test.jpg"`

### Q3：模型下载太慢
**解决方法**：
- 模型会在第一次运行时自动下载
- 如果下载失败，可以手动从官网下载后放在指定位置
- YOLOv8模型下载地址：https://github.com/ultralytics/assets

### Q4：如何使用GPU加速
**前提条件**：
- 需要有 NVIDIA 显卡
- 需要安装 CUDA 工具包

**检查是否支持GPU**：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
如果输出 `True`，说明可以使用GPU。

**使用GPU运行**：
```bash
python yolov8n_detector.py --image test.jpg --device cuda
```

### Q5：结果图片保存在哪里
默认保存在代码所在文件夹，文件名为 `output.jpg`。

可以用 `--output` 参数指定保存路径：
```bash
python yolov8n_detector.py --image test.jpg --output my_result.jpg
```

### Q6：如何处理视频
```bash
# 处理视频文件
python yolov8n_detector.py --video input.mp4 --output output.mp4

# 处理摄像头（需要摄像头设备）
python yolov8n_detector.py --video 0 --show
```

### Q7：Web应用无法启动
**错误信息**：`ModuleNotFoundError: No module named 'flask'`

**解决方法**：
```bash
pip install flask flask-cors -q
```

### Q8：Web应用上传图片失败
**可能原因**：
1. 图片格式不支持（支持：png、jpg、jpeg、gif、bmp）
2. 图片文件过大
3. 服务器没有正常启动

---

## 7️⃣ 示例输出解读

### 目标检测输出示例

运行命令后，会看到类似这样的输出：
```
2024-01-15 10:30:00 - INFO - 模型加载成功，耗时: 2.50秒
2024-01-15 10:30:01 - INFO - 检测结果已保存到: output.jpg
```

打开 `output.jpg`，会看到：
- 🟢 绿色框框：检测到的物体
- 框上面的文字：物体名称和置信度（比如 `person: 0.95`）

### 图像分类输出示例

运行命令后，会看到类似这样的输出：
```
分类结果 (Top-5):
1. golden_retriever: 0.9234
2. labrador_retriever: 0.0512
3. cocker_spaniel: 0.0123
4. german_shepherd: 0.0089
5. husky: 0.0042
```

解读：
- 第一行 `golden_retriever: 0.9234` 表示：最可能是金毛犬，置信度92.34%
- 数值越大，置信度越高

### Web应用输出示例

上传图片后，会在页面下方看到：
- 🖼️ 处理后的图片，带有检测框
- 🏷️ 检测到的物体列表和置信度
- 🔗 如果启用分组，会按类别分组显示

---

## 🎯 新手练习任务

### 任务1：使用Web应用检测图片
1. 打开浏览器访问 http://127.0.0.1:5000
2. 找一张包含人物或动物的图片
3. 上传图片并选择「目标检测」
4. 观察检测结果，尝试启用「按类别分组」

### 任务2：使用命令行分类图片
1. 找一张水果或动物的图片
2. 保存为 `cat.jpg`
3. 运行命令：`python efficientnetv2_classifier.py --image cat.jpg --topk 3`
4. 查看分类结果是否正确

### 任务3：比较不同模型
```bash
# 用YOLOv8n检测
python yolov8n_detector.py --image test.jpg --output result_n.jpg

# 用YOLOv8s检测
python yolov8s_detector.py --image test.jpg --output result_s.jpg

# 比较两张结果图片的差异
```

---

## 📁 文件清单

当前文件夹包含以下文件：

| 文件 | 说明 |
|------|------|
| `app.py` | Web应用后端服务 |
| `templates/index.html` | Web应用前端页面 |
| `yolov8n_detector.py` | YOLOv8n 目标检测（命令行） |
| `yolov8s_detector.py` | YOLOv8s 目标检测（命令行） |
| `mobilenetv4_classifier.py` | MobileNetV4 图像分类（命令行） |
| `efficientnetv2_classifier.py` | EfficientNetV2 图像分类（命令行） |
| `models_introduction.md` | 模型详细介绍 |
| `新手使用手册.md` | 本手册 |
| `uploads/` | 上传文件临时目录 |
| `outputs/` | 处理结果保存目录 |

---

## 💡 小提示

1. **第一次运行会下载模型**，可能需要几分钟，请耐心等待
2. **模型文件较大**（几十MB），建议在网络良好时运行
3. **CPU运行较慢**是正常的，有GPU会快很多
4. **Web应用更方便**，推荐初学者使用Web界面
5. 如果遇到问题，可以先查看命令输出的错误信息，通常会提示解决方案

---

## 📞 遇到问题怎么办

1. 先仔细阅读错误信息
2. 检查文件名和路径是否正确
3. 确保所有依赖都已安装
4. 尝试重新安装依赖：`pip install ultralytics torch timm opencv-python flask --upgrade`

---

🎉 **恭喜你！** 现在你已经学会了使用四种深度学习模型和Web应用！快找些图片来试试吧！

---

*手册版本：2.0*  
*更新日期：2026年5月*
