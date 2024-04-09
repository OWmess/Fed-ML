import os
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from collections import Counter

def save_images(images, labels, fold):
    os.makedirs(f"fold_{fold}", exist_ok=True)
    for i, (image, label) in enumerate(zip(images, labels)):
        im = transforms.ToPILImage()(image)
        if not os.path.exists(f"fold_{fold}/{label}"):
            os.makedirs(f"fold_{fold}/{label}")
        im.save(f"fold_{fold}/{label}/image_{i}.png")

# 输出各个label的数量
def check_fold(fold):
    for label in range(10):
        print(f"fold {fold}, label {label}: {len(os.listdir(f'fold_{fold}/{label}'))}")
    pass


# 加载MNIST数据集
mnist_train = dsets.MNIST(root='./data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

images = mnist_train.data
labels = mnist_train.targets

# 使用StratifiedKFold进行分层抽样，确保每个分组中每个类别的比例相同
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (_, test_index) in enumerate(skf.split(images, labels)):
    X_fold, y_fold = images[test_index], labels[test_index]
    save_images(X_fold, y_fold, fold)
    check_fold(fold)


