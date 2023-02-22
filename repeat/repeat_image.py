import matplotlib.pyplot as plt
import numpy as np
from ourmodel_image import ourmodel 
import os
import time
#读取imagenet32数据
def read_npz():
    with np.load('/home/zhaobenyan/robustness/datas/imagenet32.npz') as f:
        labels=f['labels']
        data=f['data']
    return labels,data

#判断路径是否存在，不存在则创建
def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

#展示并保存图片 dir:保存数据路径/home/zhaobenyan/dataset/output/imagenet32
def show_image(dir):
    labels,data=read_npz()
    for i in range(1,11):#第10、20...100张图
        label=labels[10*i]
        image=data[10*i]
        image =image.reshape(3,32,32)
        image = image.transpose(1,2,0)
        dir_image=dir+'/image{}'.format(i)
        path_exist(dir_image)
        plt.imshow(image)
        plt.savefig(os.path.join(dir_image, 'image{}'.format(i)))
        plt.close()


#得到fr dir:fr文件保存路径time:运行时间 repeat:重复次数
def get_fr(dir,time,repeat):
    labels,data=read_npz()
    for i in range(1,11):#第10、20...100张图
        label=labels[10*i]
        image=data[10*i]
        fr_list=[]
        for j in range(repeat):
            LGN_spike_time,fr=ourmodel(image)
            fr_list.append(fr)
        dir_image=dir+'/image{}'.format(i)
        np.savez(dir_image+'/fr_time{}'.format(time),fr=fr_list)

start =time.perf_counter()
if __name__ == "__main__" :
    dir='/home/zhaobenyan/dataset/output/imagenet32'
    get_fr(dir,5,10)

end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))
