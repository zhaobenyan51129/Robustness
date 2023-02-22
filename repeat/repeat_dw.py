import matplotlib.pyplot as plt
import numpy as np
from ourmodel_repeat import ourmodel 
import os


#得到fr file:输入grating/draftgrating所在路径  dir:fr文件保存路径 repeat:重复次数
def get_fr(file,dir,time,repeat):
    fr_list=[]
    for i in range(repeat):
        LGN_spike_time,fr=ourmodel(file)
        fr_list.append(fr)
    np.savez(dir+'/fr_time{}'.format(time),fr=fr_list)

#读取输出
def read_output(file_name):
    with np.load(file_name) as f:
        fr=f['fr']
    return fr

#判断路径是否存在，不存在则创建
def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

#grating/draftgrating重复实验
def experiment(time,repeat,size):
    '''
    time:模拟时长=nt*dt(这两个参数在minimal.cfg设置)
    repeat:重复次数，设置为10次
    size:grating大小
    '''
    for i in range(10):
        #实验用的grating的.bin文件所在路径
        file='/home/zhaobenyan/dataset/grating/grating_{}x{}_frameRate1/static_color-grid_{}.bin'.format(size,size,i+1)
        #储存输出文件的路径
        dir='/home/zhaobenyan/dataset/output/grating_{}x{}/contrast{}/'.format(size,size,i+1)
        path_exist(dir)
        get_fr(file,dir,time,repeat)
# dir='/home/zhaobenyan/dataset/output/grating_16x16/contrast10/'         
# fr=read_output(dir+'fr_time1.npz')
# print(np.count_nonzero(fr[0]==0))
experiment(1,10,32)


