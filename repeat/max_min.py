import matplotlib.pyplot as plt
import numpy as np
import os
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

def read_output(file_name):
    with np.load(file_name) as f:
        fr=f['fr']
    return fr

#读取数据n_pic:图片数量 repeat：重复次数 dir:读取数据的目录，time:实验时间
def read_data(n_pic,repeat,dir,time):
    fr=[]
    for k in range(n_pic):
        for i in range(repeat):
            fr_vector=read_output(dir+'/contrast{}/fr_time{}.npz'.format(k+1,time))
            fr.append(fr_vector[i])
    return fr

#计算数据 储存为一个列表，长度为n_pin,列表的每一项为一个（3,3840）的array，分别储存每张图片的fr的mean、max、min（按mean升序排列）
def compute_data(n_pic,repeat,fr_vector):
    fr = np.array(fr_vector).reshape(n_pic, repeat, -1)
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
def plot_max_min(n_pic,sorted_data,time,dir_output):
    fig = plt.figure(figsize=(50,20),dpi=200)
    for i in range(n_pic):
        sorted_mean, sorted_max,sorted_min = zip(*sorted_data[i])
        ax = fig.add_subplot(2,5,i+1)
        plt.plot(np.arange(3840),sorted_mean,linewidth=0.3)
        plt.fill_between(np.arange(3840),sorted_max,sorted_min, color='r', alpha=.8, linewidth=0)
        plt.xlim([0,4000])
        plt.ylim([0,60])
        plt.title('contrast={}'.format((i+1)/2))
    plt.savefig(os.path.join(dir_output, 'repeat_time{}.png'.format(time)))#第一个是指存储路径，第二个是图片名字
    plt.close()

time=1  #运行时间
n_pic=10      #图片个数
repeat=10    #重复次数
dir='/home/zhaobenyan/dataset/output/draftgrating_32x32'    #需要读取的数据所在路径
fr_vector=read_data(n_pic,repeat,dir,time)
sorted_data_list=compute_data(n_pic,repeat,fr_vector)
plot_max_min(n_pic,sorted_data_list,time,dir)
