import torchvision
import numpy as np

# 加载CIFAR训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# 将图像数据转换为NumPy数组
train_images = np.array(trainset.data)
train_labels = np.array(trainset.targets)

# 打印数据形状
print("训练集图像形状:", train_images.shape)
print("训练集标签形状:", train_labels.shape)

# 加载CIFAR测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# 将图像数据转换为NumPy数组
test_images = np.array(testset.data)
test_labels = np.array(testset.targets)

# 打印数据形状
print("测试集图像形状:", test_images.shape)
print("测试集标签形状:", test_labels.shape)

np.save('cifar.npy', train_images)