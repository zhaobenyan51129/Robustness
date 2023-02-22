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

#将多张图片合成mxn的大图 dir:图片路径，共m*n张，dir_out:输出图片路径
def merge_images(m,n,dir,dir_out):
    images=[]
    for j in range(m):
        images_h=[]
        for i in range(j*n+1,(j+1)*n+1):      
            img = cv2.imread(dir+"/contrast{}/time_0.png".format(i))
            images_h.append(img)
        image = jigsaw(images_h,direction="horizontal", gap=1)
        images.append(image)
    img_merged= jigsaw(images,direction="vertical", gap=1)
    cv2.imwrite(dir_out+"/time_0_merged.png", img_merged)


if __name__ == '__main__':

    #merge_noise_test()
    #merge_single_neuro()
    #dir='/home/zhaobenyan/dataset/output/imagenet32'
    dir='/home/zhaobenyan/dataset/output/grating_32x32'
    dir_out='/home/zhaobenyan/dataset/output/grating_32x32/merged'
    merge_images(2,5,dir,dir_out)
