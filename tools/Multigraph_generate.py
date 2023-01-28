import torch
import torch.nn as nn
import torchvision
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from vit_model import vit_base_patch16_224
#from our_model import ourmodel

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
def generate_images(image_size,testset,IMAGENET_MEAN,IMAGENET_STD):
      num_runs = 100
      batch_file = '/home/zhaobenyan/Attack_robustness/vit_model/files/images_labels.pth'
      images = torch.zeros(num_runs, 3, image_size, image_size) # [100, 3, 224, 224]
      labels = torch.zeros(num_runs).long() # [100,][0,0,...,0]
      preds = labels + 1 # [100, ][1,1,...,1]
      while preds.ne(labels).sum() > 0: # 没全预测对则继续循环 ne:不相等返回1
            idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)] # 过滤没预测对的， .long-> torch.int64
            for i in list(idx):
                  images[i], labels[i] = testset[random.randint(0, len(testset) - 1)] # 0~49999
            preds[idx], _ = get_preds(vit_b_16, images[idx],IMAGENET_MEAN,IMAGENET_STD)
      torch.save({'images': images, 'labels': labels}, batch_file)# [100, 3, 224, 224], [100]

def main():
      data_root = '/home/zhaobenyan/img/imagenet'
      IMAGENET_MEAN = [0.485, 0.456, 0.406]
      IMAGENET_STD = [0.229, 0.224, 0.225]
      IMAGENET_TRANSFORM = torchvision.transforms.Compose([
      torchvision.transforms.Resize(256),
      torchvision.transforms.CenterCrop(224),
      torchvision.transforms.ToTensor()])
      image_size = 224
      testset = torchvision.datasets.ImageFolder(data_root + '/val', IMAGENET_TRANSFORM) # 加载了50000张测试图
      generate_images(image_size,testset,IMAGENET_MEAN,IMAGENET_STD)

main()


