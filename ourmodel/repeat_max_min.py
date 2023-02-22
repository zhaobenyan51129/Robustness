import matplotlib.pyplot as plt
import numpy as np
import time
import os
np.seterr(divide='ignore',invalid='ignore')#忽略出现0/0时的警告
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

#读取数据
def read_data(n_pic,repeat,dir):
    fr=[]
    for k in range(1,n_pic+1):
        for i in range(1,repeat+1):
            with open(dir+'sample_spikeCount_repeat_'+str(i)+'_'+str(k)+'.bin') as f:
                sampleSize = np.fromfile(f, 'u4', 1)[0] #一共5120
                sample_t0, sample_t1 = np.fromfile(f, 'f4', 2)#t0是开始时间，t1是结束时间，我一共跑了1s
                nt = np.fromfile(f, 'u4', 1)[0]
                nLGN = np.fromfile(f, 'u4', 1)[0]
                LGN_spike_time = np.fromfile(f, 'u4', nLGN*nt)
                sampleID = np.fromfile(f, 'u4', sampleSize)#id排序是顺序的
                sample_spikeCount = np.fromfile(f, 'u4', sampleSize)
                fr.append(sample_spikeCount)
    return fr

#计算数据 储存为一个列表，长度为n_pin,列表的每一项为一个（3,3840）的array，分别储存每张图片的fr的mean、max、min（按mean升序排列）
def compute_data(n_pic,repeat,nt,dt,fr_vector):
    fr = np.array(fr_vector).reshape(n_pic, repeat, -1)/nt/dt*1000
    fr_mean = np.mean(fr,axis=1)
    fr_max = np.max(fr,axis=1)
    fr_min = np.min(fr,axis=1)
    sorted_data_list=[]
    for i in range(n_pic):
        zip_mean_max_min=zip(fr_mean[i],fr_max[i],fr_min[i])
        sorted_zip = sorted(zip_mean_max_min, key=lambda x:x[0])
        sorted_data_list.append(sorted_zip)
    return  sorted_data_list

#画图
def plot_max_min(n_pic,sorted_data,dir_output):
    fig = plt.figure(figsize=(30,20),dpi=200)
    for i in range(n_pic):
        sorted_mean, sorted_max,sorted_min = zip(*sorted_data[i])
        ax = fig.add_subplot(2,3,i+1)
        plt.plot(np.arange(3840),sorted_mean,linewidth=0.3)
        plt.fill_between(np.arange(3840),sorted_max,sorted_min, color='r', alpha=.8, linewidth=0)
        plt.ylim([0,60])
    plt.savefig(os.path.join(dir_output, 'repeat_time1.png'))#第一个是指存储路径，第二个是图片名字
    plt.close()

def main():
    n_pic=6      #图片个数
    repeat=10    #重复次数
    nt = 8000
    dt = 0.125
    dir='/home/zhaobenyan/data/data_repeat/'    #需要读取的数据所在路径
    dir_output='/home/zhaobenyan/data/dw_test_new/tmp'   #储存输出图片的路径
    fr_vector=read_data(n_pic,repeat,dir)
    sorted_data_list=compute_data(n_pic,repeat,nt,dt,fr_vector)
    plot_max_min(n_pic,sorted_data_list,dir_output)
main()