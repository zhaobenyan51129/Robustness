import torch
import torch.nn as nn
import torchvision
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("/home/zhaobenyan/robustness/vitmodel") # 将vit_model所在的文件夹路径放入sys.path中
from vit_model import vit_base_patch16_224
### 获取Imagenet标签
import json
with open("/home/zhaobenyan/robustness/datas/imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i, x in json.load(f).items()}

vit_b_16 = vit_base_patch16_224()
weights_path = "/home/zhaobenyan/grad_cam/vit_base_patch16_224.pth"
vit_b_16.load_state_dict(torch.load(weights_path, map_location="cpu"))

def apply_normalization(imgs,IMAGENET_MEAN,IMAGENET_STD): # imgs: [h, w, 3] 或 [b, h, w, 3]
    """ImageNet图片喂入模型前的标准化处理"""
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    imgs_tensor = imgs.clone()
    if imgs.dim() == 3:
        for i in range(imgs_tensor.size(0)):
            imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
    else:
        for i in range(imgs_tensor.size(1)):
            imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor

def get_preds(model, X,IMAGENET_MEAN,IMAGENET_STD): # X: [b, 224, 224, 3]
    """获取model的预测""" 
    max_value_and_idx =  model(apply_normalization(X,IMAGENET_MEAN,IMAGENET_STD)).max(dim=1) ### 注意送入模型前执行标准的normalize流程
    return max_value_and_idx[1], max_value_and_idx[0] # 获得预测的label和对应概率
    # labels: [batch,] int
    # probs: [batch,] float

# 提取vit—b-16预测对了的100张ImageNet图片并保存图片，标签
def generate_images():
    num_runs = 100
    global batch_file
    batch_file = '/home/zhaobenyan/robustness/datas/images_labels_{}.pth'.format(image_size)
    images = torch.zeros(num_runs, 3, image_size, image_size) # [100, 3, 224, 224]
    labels = torch.zeros(num_runs).long() # [100,][0,0,...,0]
    preds = labels + 1 # [100, ][1,1,...,1]
    while preds.ne(labels).sum() > 0: # 没全预测对则继续循环 ne:不相等返回1
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)] # 过滤没预测对的， .long-> torch.int64
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)] # 0~49999
        preds[idx], _ = get_preds(vit_b_16, images[idx],IMAGENET_MEAN,IMAGENET_STD)
        torch.save({'images': images, 'labels': labels}, batch_file)# [100, 3, 224, 224], [100]

##载入多图攻击的100张图
def read_image():
    images_labels = torch.load(batch_file)
    images = images_labels['images']  #(torch.Size([100, 3, 224, 224])
    labels = images_labels['labels']  #torch.Size([100]))
    #imgs=images.numpy().transpose(0,2,3,1)  #(100, 224, 224, 3)
    return images,labels

def show_images(imgs, num_rows, num_cols, titles=None, suptitle=None, scale=1.5):
    """绘制图像列表, 适用于展现多个样本"""
    figsize = (num_cols * scale, (num_rows + 0.25) * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):# tensor
            ax.imshow(img.numpy())
        else:# PIL
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.suptitle(suptitle)
    return axes

if __name__ == "__main__" :
    data_root = '/home/zhaobenyan/img/imagenet'
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    image_size = 224
    IMAGENET_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(image_size),
    torchvision.transforms.ToTensor()])
    testset = torchvision.datasets.ImageFolder(data_root + '/val', IMAGENET_TRANSFORM) # 加载了50000张测试图
    generate_images()
    images,labels=read_image()
    show_images(images.permute(0, 2, 3, 1), 10, 10, titles=[imagenet_classes[int(label)] for label in labels], suptitle='Our Chosen Images and Labels', scale=1.75);
    plt.savefig('/home/zhaobenyan/robustness/datas/imagenet{}.jpg'.format(image_size))




