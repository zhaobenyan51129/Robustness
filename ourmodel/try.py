import numpy as np
import time
import os
np.seterr(divide='ignore',invalid='ignore')
from attack_dw import ljy_bin_image_to_tensor
from attack_dw import experiment

from attack_dw import read_output_contrast
from attack_dw import Plot
np.seterr(divide='ignore',invalid='ignore')#忽略出现0/0时的警告
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

mode='q'  
dir='/home/zhaobenyan/data/dw_test_new'
delta_list = np.arange(0, 0.5, 0.01)

start =time.perf_counter()
if mode == 'gray':            
      #灰度图实验
      X = 0.5 * np.ones(shape=[32 ,32, 3])
      dir_gray=dir+'/gray'
      experiment(X, dir_gray, delta_list)
      #v1,v1_error=read_output(dir_gray+'/output.npz')
      #print('v1_output:',v1)

elif mode == 'grating':
      #grating实验
      for i in range(1,11):  
            #测试用的grating所在目录                
            file_name='/home/zhaobenyan/data/resource_contrast32_new/static_color-grid_{}.bin'.format(i)
            X_grating = ljy_bin_image_to_tensor(file_name)
            dir_grating=dir+'/Contrast{}_withclip'.format(i)
            experiment(X_grating, dir_grating, delta_list)
            #v1,v1_error=read_output(dir_grating+'/output.npz')
            #print('contrast{}_v1_output:'.format(i),v1)


elif mode=='time':
      #对不同time输入重复多次实验，横坐标是time,好像需要手动修改时间生成文件之后再读取并画图
      #time:(nt=[8000,40000])
      # delta_list=np.zeros(3)
      # file_name='/home/zhaobenyan/data/resource_contrast32_new/static_color-grid_10.bin'
      # X_grating = ljy_bin_image_to_tensor(file_name)
      # dir_grating=dir+'/time1_Contrast10'
      # if not os.path.exists(dir_grating):
      #       os.mkdir(dir_grating)
      # experiment_repeat_contrast(X_grating, dir_grating, delta_list)
      v1=[]
      error_1st_time=[]
      error_2nd_time=[]
      error_3rd_time=[]
      error_list=[]
      for i in range(5):
            dir_grating=dir+'/time{}_Contrast10'.format(i+1)
            ouput_v1,error_v1=read_output_contrast(dir_grating+'/output.npz')
            v1.append(ouput_v1)
            error_1st_time.append(error_v1[0])
            error_2nd_time.append(error_v1[1])
            error_3rd_time.append(error_v1[2])
      error_list.append(error_1st_time)
      error_list.append(error_2nd_time)
      error_list.append(error_3rd_time)
      print('len(error_1st_time):',error_1st_time)
      print('len(error_list):',len(error_list))

      plot=Plot('/home/zhaobenyan/data/dw_test/images_merged',delta_list)
      plot.plot_contrast(v1,error_list)


end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))