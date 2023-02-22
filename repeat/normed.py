import matplotlib.pyplot as plt
import numpy as np
import time
import os
import collections

np.seterr(divide='ignore',invalid='ignore')#忽略出现0/0时的警告
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

#读取重复十次之后的数据
def read_output(file_name):
    with np.load(file_name) as f:
        fr=f['fr']
    return fr

#计算范数
def compute_norm(dir,time,Norm=None):
    '''
    计算fr及误差的范数，默认为二范数
    dir:grating重复十次的数据所在地址
    Norm:用什么范数，默认为None(二范数)
    1:1范数，2：2范数，np.inf：无穷范数
    '''
    fr=read_output(dir+'/fr_time{}.npz'.format(time))  #长为10的列表，第i项为第i次重复的fr向量（3840，1）
    fr_vector=np.array(fr).reshape(10, -1)  #reshape为（10，3840）
    fr_norm=np.linalg.norm(fr, ord=Norm, axis=1)   #fr的二范数（10，1）
    error=fr_vector-fr_vector[0]        #error=第i次重复的数据减去第一次的,(10,3840)
    error_norm=np.linalg.norm(error, ord=Norm, axis=1)/fr_norm[0] #error的二范数(除以第一项的二范数）（10，1）第一项必定是0
    np.savez(dir+'/output_norm_{}'.format(time),fr_norm=fr_norm,error_norm=error_norm)#储存数据

#读取计算范数之后的数据，norm文件
def read_norm(dir,time):
    with np.load(dir+'/output_norm_{}.npz'.format(time)) as f:
        fr_norm=f['fr_norm']
        error_norm=f['error_norm']
    return fr_norm, error_norm

#print('--------------------------------实验1：每一次error的值-----------------------------------------')
def plot_repeat(fr,error):
    '''
    画图1：横坐标为重复的次数（设置为10）
    纵坐标1为error的二范数，纵坐标2为fr的二范数
    '''
    x=np.arange(10)+1  #[1,2,...,10]
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(111)
    plt.plot(x, error, 'r'+'o-',label="error")
    ax1.legend(loc=1)
    plt.xlim((0,11))
    plt.ylim((np.min(error)-0.5,np.max(error)+0.5))
    ax1.set_ylabel('error');
    ax1.set_xlabel('repeat')
    for a, b in zip(x, np.around(error,2)):  # 添加这个循坏显示坐标
        plt.text(a, b, b, color = "b",ha='center', va='bottom', fontsize=10)
    ax2 = ax1.twinx() 
    ax2.plot(x, fr, 'g'+'o-',label = "fr")
    ax2.legend(loc=2)
    plt.ylim((np.min(fr)-0.5),np.max(fr)+0.5)
    ax2.set_ylabel('fr')
    for a, b in zip(x, np.around(fr,2)):  # 添加这个循坏显示坐标
        plt.text(a, b, b,color = "b", ha='center', va='bottom', fontsize=10)

def plot1(dir,time):
    for i in range(10):
        file=dir+'/contrast{}'.format(i+1)
        #compute_norm(file,time)  #计算范数，运行一次即可，后续实验可以直接读取
        fr_norm,error_norm=read_norm(file,time)
        plot_repeat(fr_norm,error_norm)
        plt.savefig(os.path.join(file, 'repeat_{}.png'.format(time)))#第一个是指存储路径，第二个是图片名字
        plt.close()


#print('-----------------------------------实验2：error随contast的变化-----------------------------------------')
def get_data_contrast(dir,time):
    error=[]
    y2=[]
    for i in range(10):
        file=dir+'/contrast{}'.format(i+1)
        fr_norm,error_norm=read_norm(file,time)#(10,1)
        error.append(error_norm)
        y2.append(fr_norm[0])
    error=np.array(error).T #(10,10)
    return error,y2

def plot_contrast(error,y2,error_mean,time):
    '''画图2：横坐标为contrast的值，
    纵坐标1为error的二范数(重复3次，数据为error除以了第0次fr二范数的数据），
    纵坐标2为第0次fr的二范数
    '''
    x=np.arange(0.05,0.55,0.05)
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(111)
    for i in range(10):
        plt.scatter(x, error[i])#label=' ')#label="{}".format(i))
        plt.plot(x,error_mean,label = "mean")
        for a, b in zip(x, np.around(error_mean,4)):  # 添加这个循坏显示坐标
            plt.text(a, b, b, color = "b",ha='center', va='bottom', fontsize=10)
    plt.xlim((0,0.6))
    plt.ylim((np.min(error),np.max(error)+0.01))
    ax1.set_ylabel('error');
    ax1.set_xlabel('contrast')
    ax2 = ax1.twinx() 
    ax2.plot(x, y2, 'r'+'o-',label = "fr[0]")
    ax2.legend(loc=4)
    plt.ylim((np.min(y2)-50),np.max(y2)+50)
    ax2.set_ylabel('fr[0]')
    for a, b in zip(x, np.around(y2,2)):  
        plt.text(a, b, b,color = "m", ha='center', va='bottom', fontsize=10)
    plt.title('time={}s'.format(time),fontsize='large',loc='left',fontweight='bold',style='italic',family='monospace')
    

def plot2(dir,time):
    error,y2=get_data_contrast(dir,time)
    #print(error)
    error_mean=np.mean(error[1:],axis=0)
    #print('error_mean',error_mean)
    plot_contrast(error,y2,error_mean,time)
    #plt.savefig(os.path.join(dir+'/merged', 'contrast_time{}s.png'.format(time)))#第一个是指存储路径，第二个是图片名字
    plt.savefig(os.path.join(dir+'/merged', 'contrast_noseed.png'))
    plt.close()

#print('----------------------------实验3：error随时间的变化-----------------------------')
def get_data_time(dir,Norm=None):
    error=[]
    error_mean=[]
    y2=[]
    for i in range(1,6):
        fr=read_output(dir+'/fr_time{}.npz'.format(i))  #长为10的列表，第i项为第i次重复的fr向量（3840，1）
        fr_vector=np.array(fr).reshape(10, -1)  #reshape为（10，3840）
        fr_norm=np.linalg.norm(fr, ord=Norm, axis=1)   #fr的二范数（10，1）
        error_vector=fr_vector-fr_vector[0]        #error=第i次重复的数据减去第一次的,(10,3840)
        error_norm=np.linalg.norm(error_vector, ord=Norm, axis=1)/fr_norm[0] #error的二范数(除以第一项的二范数）（10，1）第一项必定是0
        error_mean_norm=np.mean(error_norm[1:])
        error.append(error_norm)
        error_mean.append(error_mean_norm)
        y2.append(fr_norm[0])
    error=np.array(error).T
    return error,y2,error_mean

def plot_time(error,y2,error_mean,contrast):
    '''画图3横坐标为时间（单位s), 
    纵坐标1为error的二范数(重复三次），纵坐标2为第0次fr的二范数
    '''
    x=[1,2,3,4,5]  #横坐标：时间
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(111)
    for i in range(10):
        plt.scatter(x, error[i])#,label="{}".format(i+1))
        plt.plot(x,error_mean,label = "mean")
        for a, b in zip(x, np.around(error_mean,4)):  # 添加这个循坏显示坐标
            plt.text(a, b, b, color = "b",ha='center', va='bottom', fontsize=10)
    plt.xlim((0,5.5))
    plt.ylim((0,np.max(error)+0.005))
    ax1.set_ylabel('error');
    ax1.set_xlabel('time/s')
    ax2 = ax1.twinx() 
    ax2.plot(x, y2, 'r'+'o-',label = "norm2(fr) for the 0th time")
    ax2.legend(loc=4)
    plt.ylim(0,np.max(y2)*1.1)
    ax2.set_ylabel('norm2(fr) for the 0th time')
    for a, b in zip(x, np.around(y2,2)):  
        plt.text(a, b, b,color = "m", ha='center', va='bottom', fontsize=10)
    plt.title('contrast={}'.format((contrast+1)/20),fontsize='large',loc='left',fontweight='bold',style='italic',family='monospace')

def plot3(dir):
    for i in range(10):
        file=dir+'/contrast{}'.format(i+1)
        error,y2,error_mean=get_data_time(file)
        print(len(error_mean))
        plot_time(error,y2,error_mean,i)
        plt.savefig(os.path.join(file,'time_0.png'))#第一个是指存储路径，第二个是图片名字
        plt.close()

#print('--------------------------------实验4，error的分布---------------------------')
def plot_error_distribution(dir,time):
    fig = plt.figure(figsize=(50,20),dpi=200)
    for i in range(10):
        file=dir+'/contrast{}'.format(i+1)
        fr=read_output(file+'/fr_time{}.npz'.format(time))  #长为10的列表，第i项为第i次重复的fr向量（3840，1）
        fr_vector=np.array(fr).reshape(10, -1)  #reshape为（10，3840）
        error=fr_vector-fr_vector[0]        #error=第i次重复的数据减去第一次的,(10,3840)
        error_mean = abs(np.mean(error[1:,:],axis=0))
        error_count=collections.Counter(error_mean)
        #print('error_count:',error_count)
        ax = fig.add_subplot(2,5,i+1)
        frequency_each, _1, _2 =plt.hist(error_mean,bins=np.arange(0,np.max(error_mean)+0.1,0.1),rwidth=0.7)
        #print(frequency_each)
        for a, b in zip(np.arange(0,np.max(error_mean)+0.1,0.1), np.around(frequency_each,3)):  # 添加这个循坏显示坐标
            plt.text(a, b, b, color = "r",ha='left', va='bottom', fontsize=10)
        plt.title('contrast={}'.format((i+1)/20),fontsize='large',loc='left',fontweight='bold',style='italic',family='monospace')
    plt.suptitle('time={}s'.format(time), ha = 'left',fontsize = 30,weight = 'extra bold')
    plt.savefig(os.path.join(dir+'/merged', 'error_distribution_time{}.png'.format(time)))#第一个是指存储路径，第二个是图片名字
    plt.close()

#start =time.perf_counter()
if __name__ == "__main__" :
    #计算noseed的范数
    # dir_noseed='/home/zhaobenyan/dataset/output/grating_32x32_noseed'
    # for i in range(10):
    #     dir=dir_noseed+'/contrast{}'.format(i+1)
    #     compute_norm(dir,1)
    # dir_noseed='/home/zhaobenyan/dataset/output/grating_32x32_noseed'
    # plot2(dir_noseed,1)
    # for i in range(5):
    #     time=i+1
    #     #plot_error_distribution(dir,time)
    #     # plot1(dir,time)
    #     plot2(dir,time)’
    dir='/home/zhaobenyan/dataset/output/grating_32x32'
    plot3(dir)
    
# end = time.perf_counter()
# print('Running time: %s Seconds'%(end-start))
