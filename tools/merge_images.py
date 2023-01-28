# coding=utf-8
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def jigsaw(imgs, direction="vertical", gap=0):
    imgs = [Image.fromarray(img) for img in imgs]
    w, h = imgs[0].size
    if direction == "horizontal":#水平
        result = Image.new(imgs[0].mode, ((w+gap)*len(imgs)-gap, h))
        for i, img in enumerate(imgs):
            result.paste(img, box=((w+gap)*i, 0))
    elif direction == "vertical":#竖直排成一列
        result = Image.new(imgs[0].mode, (w, (h+gap)*len(imgs)-gap))
        for i, img in enumerate(imgs):
            result.paste(img, box=(0, (h+gap)*i))
    else:
        raise ValueError("The direction parameter has only two options: horizontal and vertical")
    return np.array(result)

def merge_noise_test():
    #将/home/zhaobenyan/Attack_robustness/ourmodel/Noise_test.py生成的图片画成4X3的一张图
    images=[]
    img=cv2.imread('/home/zhaobenyan/data/dw_test_new/Noise_test/gray/gauss_attack.png')
    images.append(img)
    for i in range(1,11):      
        img = cv2.imread("/home/zhaobenyan/data/dw_test_new/Noise_test/Contrast{}/gauss_attack.png".format(i))
        images.append(img)
    img_horizontal=[]
    img_merged1 = jigsaw(images[0:3],direction="horizontal")
    img_merged2 = jigsaw(images[3:6],direction="horizontal")
    img_merged3 = jigsaw(images[6:9],direction="horizontal")
    img_merged4 = jigsaw(images[9:11],direction="horizontal")
    img_horizontal.append(img_merged1)
    img_horizontal.append(img_merged2)
    img_horizontal.append(img_merged3)
    img_horizontal.append(img_merged4)
    img_merged=jigsaw(img_horizontal)
    cv2.imwrite("/home/zhaobenyan/data/dw_test_new/images_merged/merged_gauss_attack_try.png", img_merged)


def merge_single_neuro():
    #将/home/zhaobenyan/Attack_robustness/ourmodel/Repeatability_single_neuro.py生成的图片纵向合并为一张
    images=[]
    for i in range(1,11):      
        img = cv2.imread("/home/zhaobenyan/data/dw_test_new/Repeatability_contrast/Contrast{}/repeat_single_neuro.png".format(i))
        images.append(img)
    
    img_merged = jigsaw(images)
    cv2.imwrite("/home/zhaobenyan/data/dw_test_new/images_merged/merged_single_neuro.png", img_merged)

def show_grating():
    #展示不同contrast的grating
    images=[]
    for i in range(1,11):      
        img = cv2.imread("/home/zhaobenyan/data/resource_contrast32_new/static_color-grid_{}_1.png".format(i))
        images.append(img)
    img_horizontal=[]
    img_merged1 = jigsaw(images[0:5],direction="horizontal", gap=1)
    img_merged2 = jigsaw(images[5:10],direction="horizontal", gap=1)
    img_horizontal.append(img_merged1)
    img_horizontal.append(img_merged2)
    img_merged=jigsaw(img_horizontal,direction="vertical", gap=1)
    cv2.imwrite("/home/zhaobenyan/data/dw_test_new/images_merged/grating2x5.png", img_merged)

if __name__ == '__main__':
    #merge_noise_test()
    #merge_single_neuro()
    show_grating()
