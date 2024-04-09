import os
import numpy as np
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from collections import Counter

def save_images(images, labels):
    folder_map = {0: '0', 1: '0', 2: '1', 3: '1', 4: '2', 5: '2', 6: '3', 7: '3', 8: '4', 9: '4'}
    for fold in range(5):
        if not os.path.exists(f"fold_{fold}"):
            os.makedirs(f"fold_{fold}")
    for i, (image, label) in enumerate(zip(images, labels)):
        im = transforms.ToPILImage()(image)
        im.save(f"fold_{folder_map[label.item()]}/image_{i}.png")


# 加载MNIST数据集
mnist_train = dsets.MNIST(root='./data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

images = mnist_train.data
labels = mnist_train.targets

save_images(images, labels)

