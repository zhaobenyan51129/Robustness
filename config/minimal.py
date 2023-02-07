import numpy as np
import matplotlib.pyplot as plt
import os
import time
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

src_dir='/home/zhaobenyan/repos/patchV1/src/'
patchfast='/home/zhaobenyan/dataset/patchfast/'

#将minimal.cfg中的替死鬼文件中的数据换成实验输入的，dir:grating所在地址 size:图片大小
def replace(dir,size):
    with open(dir) as f:
        x_1 = np.fromfile(f, 'i4', 1)
        x_2 = np.fromfile(f, 'i4', 3)
        x_3 = np.fromfile(f, 'f4', 3)
        x_4 = np.fromfile(f, 'f4', 2)
        x_5 = np.fromfile(f, 'u4', 1)
        x_6 = np.fromfile(f, 'f4', size*size*3)
        #写文件,'wb'表示覆盖写文件
        f1 = open('/home/zhaobenyan/dataset/resource/static_color-grid_0.bin', 'wb') 
        np.array(x_1).astype('i4').tofile(f1) 
        np.array(x_2, dtype='i4').tofile(f1)
        x_3.astype('f4').tofile(f1) # init_luminance
        np.array(x_4, dtype='f4').tofile(f1)
        np.array(x_5).astype('u4').tofile(f1)
        np.array(x_6).astype('f4').tofile(f1)
        f1.close() #记得要关文件,如果是用with打开是会自动关闭的

#运行patchfast，生成文件储存在patchfast
def patch():
    os.system(f'cd {patchfast} && patch_fast -c {src_dir}minimal.cfg') 

#读取输出
def read_spike(file):
    with open(file) as f:
        sampleSize = np.fromfile(f, 'u4', 1)[0] #一共5120
        sample_t0, sample_t1 = np.fromfile(f, 'f4', 2)#t0是开始时间，t1是结束时间，我一共跑了1s
        nt = np.fromfile(f, 'u4', 1)[0]
        nLGN = np.fromfile(f, 'u4', 1)[0]
        LGN_spike_time = np.fromfile(f, 'u4', nLGN*nt)
        sampleID = np.fromfile(f, 'u4', sampleSize)#id排序是顺序的
        sample_spikeCount = np.fromfile(f, 'u4', sampleSize)
        fr = sample_spikeCount/(sample_t1-sample_t0)*1000
    LGN_spike_time = LGN_spike_time.reshape((nt,nLGN)).T
    return sampleID,fr

#画出fr的分布
def plot_fr(fr,dir_output):
    plt.hist(fr,bins=np.arange(0,50,1),rwidth=0.7)
    plt.savefig(os.path.join(dir_output, 'draftgrating_3.png'))#第一个是指存储路径，第二个是图片名字
    plt.close()

#判断文件夹是否存在，如果文件夹不存在，则创建该文件夹
def path_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

start =time.perf_counter()
if __name__ == "__main__" :
    size=32
    frameRate=96 #=1:grating, >1:draftgrating, 要与minimal.cfg中对应的该参数的值一致
    for i in range(6):
        dir='/home/zhaobenyan/dataset/grating/grating_{}x{}_frameRate{}/static_color-grid_{}.bin'.format(size,size,frameRate,i+1)
        replace(dir,size)
        patch()
        output='/home/zhaobenyan/dataset/patchfast/sample_spikeCount_test_1.bin'
        dir_output='/home/zhaobenyan/dataset/output/grating_{}x{}_frameRate{}'.format(size,size,frameRate)
        path_exist(dir_output)
        sampleID,fr=read_spike(output)
        print(np.count_nonzero(fr==0))
        plt.hist(fr,bins=np.arange(0,50,1),rwidth=0.7)
        plt.savefig(os.path.join(dir_output, 'draftgrating_{}.png'.format(i)))#第一个是指存储路径，第二个是图片名字
        plt.close()
    # patch()
    # output='/home/zhaobenyan/dataset/patchfast/sample_spikeCount_test_1.bin'
    # dir_output='/home/zhaobenyan/dataset/output/grating_{}x{}_frameRate{}'.format(size,size,frameRate)
    # sampleID,fr=read_spike(output)
    # print(fr)
    # print(np.count_nonzero(fr==0))
    # plt.hist(fr,bins=np.arange(0,50,1),rwidth=0.7)
    # plt.savefig(os.path.join(dir_output, 'draftgrating_32_6.png'))#第一个是指存储路径，第二个是图片名字
    # plt.close() 
end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))