# 🚗 Easy-License-Plate-Recognition (Car-eye)

[English](#english-version) | [简体中文](#简体中文)

---

## English Version

### 🧠 Introduction

**EasyLicenseIdentify** is a lightweight end-to-end license plate recognition system combining **YOLOv8** (for detection) and **CRNN** (for character recognition).  
It aims to achieve accurate plate detection and OCR recognition while maintaining high speed and model simplicity.

---

### 📁 Project Structure

```bash
EasyLicenseIdentify/
├── data/ # Data directory (empty by default, see README)
│ ├── README.md
│
├── models/ # Pretrained model weights
│ ├── best.pt
│ ├── best_crnn.pt
│
├── outputs/ # Prediction and result outputs
│
├── src/ # Source code
│ ├── detection/
│ │ ├── data.yaml
│ │ └── yolo_retrain.py
│ ├── ocr/
│ │ └── train_crnn.py
│ └── main.py
│
├── requirements.txt
└── README.md


---
```
### 📦 Datasets

To reduce repository size, datasets are **not included**.  
Please download and place them under the `data/` folder following the paths below:

#### 1️⃣ License Plate Detection Dataset (`car_dataset/`)
- Source: [CCPD2019](https://github.com/detectRecog/CCPD)
- License: MIT License

#### 2️⃣ OCR Character Recognition Dataset (`ocr_dataset/`)
- Source: [CBLPRD-330K](https://github.com/SunlifeV/CBLPRD-330k)
- License: MIT License

---

### ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/yourname/EasyLicenseIdentify.git
cd EasyLicenseIdentify

# Create environment (recommended)
conda create -n car_eye python=3.10
conda activate car_eye

# Install dependencies
pip install -r requirements.txt

```

### 🚀 Quick Start

1️⃣ Download datasets and place them properly:
```bash
EasyLicenseIdentify/data/car_dataset/
EasyLicenseIdentify/data/ocr_dataset/
```

2️⃣ Train YOLOv8 Detection Model:
```bash
python src/detection/yolo_retrain.py
```

3️⃣ Train CRNN OCR Model:
```bash
python src/ocr/train_crnn.py
```

4️⃣ Run the full pipeline:
```bash
python src/main.py
```

✅ Outputs will be saved under:
```bash
EasyLicenseIdentify/outputs/
```

### 🧩 Model Overview

| Component | Model  | Framework  |
|-----------|--------|------------|
| Detection | YOLOv8 | Ultralytics|
| OCR       | CRNN   | PyTorch    |

### 🧾 License

This project is released under the MIT License.

### 🙌 Acknowledgments

CCPD2019 Dataset

CBLPRD-330K Dataset

Ultralytics YOLOv8


## 简体中文

### 🧠 项目简介

EasyLicenseIdentify（Car-eye） 是一个轻量级端到端车牌识别系统，结合 YOLOv8（车牌检测）与 CRNN（字符识别），
实现从图像输入到车牌文字输出的完整自动识别流程。

### 📁 项目结构

```bash
EasyLicenseIdentify/
├── data/               # 数据目录（为空，仅保留说明文件）
│   ├── README.md
│
├── models/             # 模型权重文件
│   ├── best.pt
│   ├── best_crnn.pt
│
├── outputs/            # 预测与结果输出
│
├── src/                # 源代码
│   ├── detection/
│   │   ├── data.yaml
│   │   └── yolo_retrain.py
│   ├── ocr/
│   │   └── train_crnn.py
│   └── main.py
│
├── requirements.txt
└── README.md
```


### 🧠 数据集说明

为减小项目体积，原始数据集未包含在仓库中。
请根据以下来源自行下载并放置于 data/ 目录下对应文件夹：

1️⃣ 车牌检测数据集（car_dataset/）

来源：CCPD2019

许可证：MIT License

2️⃣ 车牌字符识别数据集（ocr_dataset/）

来源：CBLPRD-330K

许可证：MIT License

#### 1. 车牌检测数据集（`car_dataset/`）
- 数据来源：[CCPD2019](https://github.com/detectRecog/CCPD)
- 许可证：MIT License

#### 2. 车牌字符识别数据集（`ocr_dataset/`）
- 数据来源：[CBLPRD-330K](https://github.com/SunlifeV/CBLPRD-330k)
- 许可证：MIT License

---

### ⚙️ 环境安装

```bash
git clone https://github.com/yourname/EasyLicenseIdentify.git
cd EasyLicenseIdentify

conda create -n car_eye python=3.10
conda activate car_eye

pip install -r requirements.txt
```

### 🚀 快速开始
1. 下载数据集并放置路径：
```bash
EasyLicenseIdentify/data/car_dataset/
EasyLicenseIdentify/data/ocr_dataset/
```

2. 训练检测模型（YOLO8s）
```bash
python src/detection/yolo_retrain.py
```

3. 训练OCR识别模型（CRNN）
```bash
python src/ocr/train_crnn.py
```

4. 运行预测
```bash
python src/main.py
```

输出结果默认保存至 outputs/ 文件夹。

### 🧩 模型说明

| 组件 | 模型   | 框架        |
|------|--------|-------------|
| 检测 | YOLOv8 | Ultralytics |
| 识别 | CRNN   | PyTorch     |

### 🧾 许可证

本项目采用 MIT License。


### 🙌 致谢

CCPD2019

CBLPRD-330K

Ultralytics YOLOv8