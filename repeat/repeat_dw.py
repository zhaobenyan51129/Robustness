import matplotlib.pyplot as plt
import numpy as np
from ourmodel_repeat import ourmodel 
import os


#得到fr file:输入grating/draftgrating所在路径  dir:fr文件保存路径 repeat:重复次数
def save_fr_lgn(file,dir):
    fr_list=[]
    lgn_list=[]
    for i in range(repeat):
        LGN_spike_time,fr=ourmodel(file)
        #LGN_spike_time:[521,8000], fr:[3840,1],目前实验只涉及fr
        fr_list.append(fr)
        lgn_list.append(LGN_spike_time)
    np.savez(dir+'/lgn_time{}'.format(time),lgn=lgn_list)
    np.savez(dir+'/fr_time{}'.format(time),fr=fr_list)

#读取输出
def read_output(file_name):
    output=os.path.basename(file_name).split('.')[0].split('_')[0] #fr or lgn
    if output=='fr':
        with np.load(file_name) as f:
            fr=f['fr']
        return fr
    else :#output=='lgn'
        with np.load(file_name) as f:
            lgn=f['lgn']
        return lgn

#判断路径是否存在，不存在则创建
def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

#grating/draftgrating重复实验
if __name__ == "__main__" :
    '''
    time:模拟时长=nt*dt(这两个参数在minimal.cfg设置)
    repeat:重复次数，设置为10次
    size:grating大小
    '''
    time=5  #运行时间
    size=32 #图片尺寸
    repeat=10
    for i in range(10):
        #实验用的grating的.bin文件所在路径
        file='/home/zhaobenyan/dataset/grating/grating_{}x{}_frameRate96_phase=pi/static_color-grid_{}.bin'.format(size,size,i+1)
        #储存输出文件的路径
        dir='/home/zhaobenyan/dataset/output/driftgrating_{}x{}/contrast{}/'.format(size,size,i+1)
        path_exist(dir)
        save_fr_lgn(file,dir)


