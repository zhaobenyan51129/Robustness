import numpy as np
import matplotlib.pyplot as plt
import os
import time
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示
src_dir='/home/zhaobenyan/repos/patchV1/src/'  #minimalTC所在文件夹
patchfast='/home/zhaobenyan/dataset/patchfast/'#输出数据（读取fr的.bin文件）所在文件夹

def ourmodel(image):
    '''
    image:输入图片[size,size,3]
    '''
    if len(image.shape)==4:
        image = image[0]  
    imgsize=image.shape[0]
    print('imgsize:',imgsize)
    #对不同尺寸的图片有不同的替死鬼文件
    dir='/home/zhaobenyan/dataset/resource/static_color-grid_{}.bin'.format(imgsize)
    #step1:将minimal.cfg中的替死鬼文件中的数据换成实验输入
    with open(dir) as f:
        x_1 = np.fromfile(f, 'i4', 1)
        x_2 = np.fromfile(f, 'i4', 3)
        nFrame=x_2[0]
        size=x_2[1]
        assert nFrame==1,'请输入静态图片'
        assert imgsize == size,'替死鬼图片与实验图片尺寸不匹配，请重新选择替死鬼文件'
        x_3 = np.fromfile(f, 'f4', 3)
        x_4 = np.fromfile(f, 'f4', 2)
        x_5 = np.fromfile(f, 'u4', 1)
        x_6 = np.fromfile(f, 'f4', nFrame*size*size*3)
        #写文件,'wb'表示覆盖写文件 f1打开“替死鬼”文件地址
        f1 = open('/home/zhaobenyan/dataset/resource/static_color-grid_{}.bin'.format(size), 'wb') 
        np.array(x_1).astype('i4').tofile(f1) 
        np.array((x_2[0],size,size), dtype='i4').tofile(f1)
        x_3.astype('f4').tofile(f1) # init_luminance
        np.array(x_4, dtype='f4').tofile(f1)
        np.array(x_5).astype('u4').tofile(f1)
        y = image.transpose([2,0,1]).flatten()
        y.astype('f4').tofile(f1)
        f1.close() #记得要关文件
    #step2 运行minimalTC并记录运行时间
    t1 = time.time()
    os.system(f'cd {src_dir} && ./minimalTC' )
    print(time.time()-t1) 
    #step3 读取输出
    with open('/home/zhaobenyan/dataset/patchfast/sample_spikeCount_test_1.bin') as f:
        sampleSize = np.fromfile(f, 'u4', 1)[0] #一共5120
        sample_t0, sample_t1 = np.fromfile(f, 'f4', 2)#t0是开始时间，t1是结束时间
        nt = np.fromfile(f, 'u4', 1)[0]
        nLGN = np.fromfile(f, 'u4', 1)[0]
        LGN_spike_time = np.fromfile(f, 'u4', nLGN*nt)
        sampleID = np.fromfile(f, 'u4', sampleSize)#id排序是顺序的
        sample_spikeCount = np.fromfile(f, 'u4', sampleSize)
        fr = sample_spikeCount/(sample_t1-sample_t0)*1000
    LGN_spike_time = LGN_spike_time.reshape((nt,nLGN)).T
    
    return LGN_spike_time, fr