import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

#读取grating的数据  file:文件路径 static_color-grid_{}_cfg.bin类型的文件
def read_grating_cfg(file):
    with open(file) as f: #可以去查查with是干嘛的，它是以防你忘记关掉文件，这种表达会比较好
        sf = np.fromfile(f,'f4',1)[0] #为啥是np.fromfile（。。。）[0],因为用np命令的读出来是array，其实我们只是要他的元素
        ori = np.fromfile(f,'f4',1)[0]
        phase = np.fromfile(f,'f4',1)[0]
        contrast = np.fromfile(f,'f4',1)[0] #一般读取到contrast就停了
        #crest = np.fromfile(f,'f4',3)  #波峰
        #valley = np.fromfile(f,'f4',3) #波谷
    return sf, ori, phase, contrast

#读取grating的数据  file:文件路径 static_color-grid_{}.bin类型的文件
def read_grating_bin(file):
    with open(file) as f:
        x_1 = np.fromfile(f, 'i4', 1)
        x_2 = np.fromfile(f, 'i4', 3)
        nFrame=x_2[0]
        size=x_2[1]
        x_3 = np.fromfile(f, 'f4', 3)
        x_4 = np.fromfile(f, 'f4', 2)
        x_5 = np.fromfile(f, 'u4', 1)
        x_6 = np.fromfile(f, 'f4', nFrame*size*size*3)
    #return nFrame,size
    return x_1,x_2,x_3,x_4,x_5,x_6

#读取grating的数据  file:文件路径 static_color-grid_{}.bin类型的文件.并转化为(size,size,3)大小的数组
#只适用于nFrame=1的grating
def image_to_tensor(file):
    with open(file) as f:
        x_1 = np.fromfile(f, 'i4', 1)
        x_2 = np.fromfile(f, 'i4', 3)
        nFrame=x_2[0]
        size=x_2[1]
        x_3 = np.fromfile(f, 'f4', 3)
        x_4 = np.fromfile(f, 'f4', 2)
        x_5 = np.fromfile(f, 'u4', 1)
        x_6 = np.fromfile(f, 'f4', nFrame*size*size*3)
        image_numpy = x_6.reshape(3,size, size)
    return image_numpy.transpose([1,2,0])


#读取_spikeCount...bin文件，file：文件路径
def read_spike(file):
    with open(file) as f:
        sampleSize = np.fromfile(f, 'u4', 1)[0] #神经元的数量
        sample_t0, t1 = np.fromfile(f, 'f4', 2) #模拟的时间(毫秒)，fr = sample_spikeCount /(t1-t0)*1000
        sampleID = np.fromfile(f, 'u4', sampleSize)
        sample_spikeCount = np.fromfile(f, 'u4', sampleSize) # 模拟时间内，sample出的神经元的spike次数
        sample_spikeCount = sample_spikeCount.astype(np.float32)
        sample_input_spikeCount = np.fromfile(f, 'f4', sampleSize) # F0，已经处理过，不需要除时间
        sample_input_spikeCount = sample_input_spikeCount.astype(np.float32)
        fr = sample_spikeCount/(t1-sample_t0)*1000 
    return sampleID, fr # [num_neurons,], [num_neurons,]

def read_spike2(file):
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

if __name__ == "__main__" :
    file_spike='/home/zhaobenyan/dataset/patchfast/sample_spikeCount_merge_test_1.bin'