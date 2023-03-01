import matplotlib.pyplot as plt
import numpy as np
from ourmodel_attack import ourmodel 
from repeat_dw import read_output
import os
import time
import json
import torch
with open("/home/zhaobenyan/robustness/datas/imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i, x in json.load(f).items()}

#print('--------------------直接读取32x32的image--------------------')
#读取imagenet32数据
# def read_npz():
#     with np.load('/home/zhaobenyan/robustness/datas/imagenet32.npz') as f:
#         labels=f['labels']
#         data=f['data']
#     return labels,data
#判断路径是否存在，不存在则创建
def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
# #展示并保存图片 dir:保存数据路径/home/zhaobenyan/dataset/output/imagenet32
# def show_image(dir):
#     labels,data=read_npz()
#     for i in range(1,11):#第10、20...100张图
#         label=labels[10*i]
#         image=data[10*i]
#         image =image.reshape(3,32,32)
#         image = image.transpose(1,2,0)
#         dir_image=dir+'/image{}'.format(i)
#         path_exist(dir_image)
#         plt.imshow(image)
#         plt.savefig(os.path.join(dir_image, 'image{}'.format(i)))
#         plt.close()
# #得到fr dir:fr文件保存路径time:运行时间 repeat:重复次数
# def save_fr_lgn(dir,time,repeat):
#     labels,data=read_npz()
#     for i in range(1,11):#第10、20...100张图
#         label=labels[10*i]
#         image=data[10*i]
#         lgn_list=[]
#         fr_list=[]
#         for j in range(repeat):
#             LGN_spike_time,fr=ourmodel(image)
#             lgn_list.append(LGN_spike_time)
#             fr_list.append(fr)
#         dir_image=dir+'/image{}'.format(i)
#         np.savez(dir_image+'/fr_time{}'.format(time),fr=fr_list)
#         np.savez(dir_image+'/lgn_time{}'.format(time),lgn=lgn_list)

print('-----------------从256x256的image中截取32x32的image----------------')
#载入多图攻击的100张图并截取32x32
def read_image():
    batch_file = '/home/zhaobenyan/robustness/datas/images_labels_224.pth'
    images_labels = torch.load(batch_file)
    images = images_labels['images']  #(torch.Size([100, 3, 224, 224])
    labels = images_labels['labels']  #torch.Size([100]))
    images=images.numpy().transpose(0,2,3,1)  #(100, 224, 224, 3)
    images_unclipped=images[90:100,:,:,:]  #前十张
    images_clipped=images[90:100,96:128,96:128,:]  #前十张，在中间截取32x32（10，32，32，3）
    return images_unclipped,images_clipped,labels[90:100]

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

#选最后10张，按照戴老师的方式计算contrast并保存图片
def save_fig():
    images_unclipped,images_clipped,labels=read_image()  #(10,32,32,3)
    x=images_clipped.reshape([10,-1]) #(10,3072)
    contrast=(np.max(x,axis=1)-np.mean(x,axis=1))/np.mean(x,axis=1)/2  #(10,)
    show_images(images_clipped, 2, 5,titles=[str(c) for c in contrast] , suptitle='Our Chosen Images and Labels', scale=1.75);
    plt.savefig(dir+'/images_clipped.jpg')
    plt.close()
    show_images(images_unclipped, 2, 5,titles=[imagenet_classes[int(label)] for label in labels] , suptitle='Our Chosen Images and Labels', scale=1.75);
    plt.savefig(dir+'/images_unclipped.jpg')
    plt.close()

#得到十张图的fr和lgn输出
def save_fr_lgn(images_clipped):
    x=images_clipped.reshape([10,-1]) #(10,3072)
    contrast=(np.max(x,axis=1)-np.mean(x,axis=1))/np.mean(x,axis=1)/2  #(10,)
    lgn_list=[]
    fr_list=[]
    for i in range(images_clipped.shape[0]):
        LGN_spike_time,fr=ourmodel(images_clipped[i])
        lgn_list.append(LGN_spike_time)
        fr_list.append(fr)
    np.savez(dir+'/fr_time1',fr=fr_list)
    np.savez(dir+'/lgn_time1',lgn=lgn_list)
    np.savez(dir+'/contrast',contrast=contrast)
    

#画出每张图fr的分布和lgn发放
def plot_fr_lgn():
    fr_list=read_output(dir+'/fr_time1.npz')
    lgn_list=read_output(dir+'/lgn_time1.npz')
    with np.load(dir+'/contrast.npz') as f:
        contrast=f['contrast']
    fig = plt.figure(figsize=(50,40),dpi=200)
    for i in range(10):
        ax = fig.add_subplot(4,5,i+1)
        fr=fr_list[i]
        plt.hist(fr,bins=np.arange(0,50,1),rwidth=0.7)
        plt.title('contrast={}'.format(contrast[i]),fontsize='xx-large',loc='left',fontweight='bold',style='italic',family='monospace')
        
        ax = fig.add_subplot(4,5,i+11)
        ar,num=np.unique(lgn_list[i],return_counts=True)
        explode = (0, 0.1)  # only "explode" the 1st slice (i.e. '0')
        numbers=[num[0],np.sum(num[1:])]
        labels=['zero','nonzero']
        ax.pie(numbers, labels=labels,explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90)
        ax.axis('equal') 
    plt.suptitle('distribution of fr and lgn_spike', ha = 'left',fontsize = 30, weight = 'extra bold')
    plt.savefig(os.path.join(dir, 'distribution of fr and lgn_spike'))#第一个是指存储路径，第二个是图片名字
    plt.close()




# start =time.perf_counter()
if __name__ == "__main__" :
    time=1
    dir='/home/zhaobenyan/dataset/output/imagenet'
    path_exist(dir)
    # images_unclipped,images_clipped,labels=read_image()  #(10,32,32,3)
    # lgn_list=read_output(dir+'/lgn_time1.npz')
    # print(lgn_list[0].shape)
    # # lgn_spike=np.array(lgn_list)  #(10,512,8000)
    # # print(lgn_spike.shape)
    # ar,num=np.unique(lgn_list[0],return_counts=True)
    # print(ar)
    # print(num)
    # print(ar[1:])
    # print([num[0],np.sum(num[1:])])
    # print([num[0],np.sum(num[1:])])
    # contrast=save_fig() 
    # save_fr_lgn(images_clipped)
    plot_fr_lgn()
   

# end = time.perf_counter()
# print('Running time: %s Seconds'%(end-start))
