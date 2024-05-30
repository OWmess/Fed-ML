# 联邦学习项目

该项目实现了一个简单的联邦学习系统，使用 LeNet-5 模型来对 MNIST 图像进行分类。系统由一个服务器和多个客户端组成，它们协同训练一个全局模型，同时保留各自的本地数据隐私。

## 目录
- [环境要求](#环境要求)
- [安装](#安装)
- [使用方法](#使用方法)
  - [服务器](#服务器)
  - [客户端](#客户端)
  - [数字识别应用](#数字识别应用)
- [项目结构](#项目结构)
- [实现细节](#实现细节)
- [参考文献](#参考文献)

## 环境要求

- torch
- torchvision
- opencv-python
- numpy
- pillow
- onnxruntime
- scikit-image
- scikit-learn
- visdom
- matplotlib

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/OWmess/Fed-ML.git
   cd Fed-ML
   ```

2. 安装所需的包：
   ```bash
   pip install -r requirements.txt
   ```

3. 启动 Visdom 服务器（用于可视化）：
   ```bash
   visdom
   ```
## 视频演示
### 训练

<video controls>
  <source src="./video/train.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### 识别

<video controls>
  <source src="./video/recognition.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 使用方法

### 服务器

1. 运行服务器脚本：
   ```bash
   python server.py
   ```
   服务器ip需在`server.py`中指明

### 客户端

1. 使用所需的参数运行客户端脚本：
   ```bash
   python client.py --client_num <CLIENT_NUMBER> --ip <SERVER_IP>
   ```

   例如，要运行第一个客户端：
   ```bash
   python client.py --client_num 1 --ip 192.168.28.220
   ```
### 数字识别应用

```bash
        python server.py
```

## 项目结构

```plaintext
federated-learning-project/
│
├── client.py          # 客户端脚本
├── server.py          # 服务器脚本
├── lenet5.py          # LeNet-5 模型定义和训练/测试函数
├── requirements.txt   # 所需 Python 包列表
├── README.md          # 项目文档
└── tools/             # 数据集文件夹
    ├── iid/           # IID 数据分区
    └── non-iid/       # 非 IID 数据分区
```

## 实现细节

### 服务器 (`server.py`)

服务器负责：
- 接受来自多个客户端的连接。
- 接收客户端发送的模型更新。
- 使用联邦平均算法聚合接收的模型。
- 将聚合后的全局模型发送回客户端。
- 使用 Visdom 可视化训练进度。



### 客户端 (`client.py`)

客户端负责：
- 加载本地的 MNIST 数据。
- 接收来自服务器的全局模型。
- 在本地数据上训练模型。
- 将更新后的模型发送回服务器。



### 模型和训练 (`lenet5.py`)

该模块定义了 LeNet-5 模型并提供了训练和测试函数。

### 数字识别应用 (`recognition.py`)

1. **图像处理**：使用 OpenCV 对图像进行预处理，包括灰度化、自适应阈值化、膨胀等操作，以便更好地提取字符。
2. **字符分割**：通过垂直投影将字符从车牌图像中分割出来。
3. **字符识别**：使用一个经过训练的神经网络模型（MNIST）对分割出的字符进行识别。
4. **用户界面**：提供了一个交互式的界面，可以实时显示处理过程，并通过滑块调整字符分割的阈值。

## 参考文献

- [LeNet-5](http://yann.lecun.com/exdb/lenet/): 一个经典的卷积神经网络架构。
- [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html): 一种在多个去中心化设备上训练机器学习模型的方法。
- [Visdom](https://github.com/facebookresearch/visdom): 一个用于实时可视化训练指标的工具。

