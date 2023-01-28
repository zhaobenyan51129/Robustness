import os
import numpy as np
import time

def ourmodel(image): # [1, 16, 16, 3] numpy
    if len(image.shape)==4:
        image = image[0]   
    ########################################################
    #static_color_1.bin是戴老师模型的输入文件，是需要x1~x6这六个参数，其中x6表示的是图像
    # 所以我们读取x1~x6，然后把x6替换成我们想要测试的图片即可 
    ########################################################
    #读文件
    with open('/home/zhaobenyan/data/resource_tmp/static_color_1.bin') as f:
        x_1 = np.fromfile(f, 'i4', 1)
        x_2 = np.fromfile(f, 'i4', 3)
        x_3 = np.fromfile(f, 'f4', 3)
        x_4 = np.fromfile(f, 'f4', 2)
        x_5 = np.fromfile(f, 'u4', 1)
        x_6 = np.fromfile(f, 'f4', 32*32*3)
        #写文件,'wb'表示覆盖写文件
        f1 = open('/home/zhaobenyan/data/resource_tmp/static_color_1' + '.bin', 'wb') 
        np.array(x_1).astype('i4').tofile(f1) 
        np.array(x_2, dtype='i4').tofile(f1)
        x_3.astype('f4').tofile(f1) # init_luminance
        np.array(x_4, dtype='f4').tofile(f1)
        np.array(x_5).astype('u4').tofile(f1)
        y = image
        y.astype('f4').tofile(f1)
        f1.close() #记得要关文件,如果是用with打开是会自动关闭的
    ########################################################
    #运行戴老师的代码
    t1 = time.time()
    os.system('cd /home/zhaobenyan/repos/patchV1/src/tmp && ./minimalTC_tmp ')
    print(time.time()-t1) #记录戴老师文件的运行时间
    ########################################################
    #这个fdr是数据文件保存的文件夹的地址,这个地址应该和minimalTC_tmp中的data_fdr保持一致
    fdr = '/home/zhaobenyan/data/resource_tmp/'
    #这个fdr是数据文件的后缀,这个后缀应该和minimalTC_tmp中的trial_suffix保持一致(差了一个下划线)
    suffix = '_tmp' 
    #fn是数据文件的地址
    fn = fdr + 'sample_OutAndInputCount' +  suffix + '_1.bin' 
    with open(fn) as f:
        sampleSize = np.fromfile(f, 'u4', 1)[0] #神经元的数量
        sample_t0, t1 = np.fromfile(f, 'f4', 2) #模拟的时间(毫秒)，fr = sample_spikeCount /(t1-t0)*1000
        sampleID = np.fromfile(f, 'u4', sampleSize)
        sample_spikeCount = np.fromfile(f, 'u4', sampleSize) # 模拟时间内，sample出的神经元的spike次数
        sample_spikeCount = sample_spikeCount.astype(np.float32)
        sample_input_spikeCount = np.fromfile(f, 'f4', sampleSize) # F0，已经处理过，不需要除时间
        sample_input_spikeCount = sample_input_spikeCount.astype(np.float32)
        fr = sample_spikeCount/(t1-sample_t0)*1000 
    
    return sample_input_spikeCount, fr # [num_neurons,], [num_neurons,]