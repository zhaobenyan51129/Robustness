#-*- coding: utf-8 -*-
from attack_dw import GaussianNoiseTestDw
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei','Times New Roman'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
import matplotlib as mpl
import numpy as np
import time
import os
np.seterr(divide='ignore',invalid='ignore')
from attack_dw import ljy_bin_image_to_tensor
np.seterr(divide='ignore',invalid='ignore')#忽略出现0/0时的警告
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示


#对不同time输入重复多次实验，横坐标是time,好像需要手动修改时间生成文件之后再读取并画图
#time:(nt=[8000,40000])
def Generate_data(delta_list,file_name,dir):
      '''
      生成并储存数据，运行一次即可，若模型有所改动则需重新生成数据
      关于时间的测试需要改变/home/zhaobenyan/repos/patchV1/src/tmp/minimal_tmp.cfg中第43行的参数nt改变时间
      每一次改变时间之后需要手动改一下储存文件的文件夹名eg:dir+'/time3_Contrast10/output_vector
      每改一次nt都要运行一次这个函数生成数据，nt越大生成数据越慢
      dalta_list:在这个实验中为全0向量
      file_name:进行测试所用grating所在路径
      dir：储存数据的路径，会生成两个文件，分别储存mode=='norm'和'vector'的数据，详见下一个函数read_data的注释
      '''
      X_grating = ljy_bin_image_to_tensor(file_name)
      dir_time=dir+'/time5_Contrast6'
      if not os.path.exists(dir_time):
            os.mkdir(dir_time)
      output=GaussianNoiseTestDw(X_grating)
      output.get_noised(delta_list)
      output.get_results()
      sampleID,unattacked_output_v1, attack_v1, error_vector_v1=output.get_vector()
      np.savez(dir_time+'/output_vector',sampleID=sampleID,unattacked_output_v1=unattacked_output_v1, attack_v1=attack_v1, error_vector_v1=error_vector_v1)#储存数据
      ouput_v1,error_v1=output.get_norm_before()
      np.savez(dir_time+'/output_norm_before',ouput_v1=ouput_v1,error_v1=error_v1)#储存数据


def merge_data(dir,mode='norm'):
      '''
      整合数据，运行一次即可，需要不同时间的数据都生成好之后再运行（这里时间设置为1-5s) 默认mode=='norm'
      mode=='norm',读取攻击之前fr的二范数（1,1)和error的二范数(delta,1),
                  然后将每个contrast的数据进行重组，得到每一次的不同contrast的fr和error
                  注意error除以了对应的fr
      mode=='vector',读取神经元编号sampleID（3066,1）攻击之前的fr（1，3066）,
                  攻击之后的fr（delta，3066），攻击之后的error（delta,3066)向量
      '''
      if mode=='norm':
            v1=[]
            error_1st_time=[]
            error_2nd_time=[]
            error_3rd_time=[]
            error_list=[]
            for i in range(5):
                  file_name=dir+'/time{}_Contrast6/output_norm_before.npz'.format(i+1)
                  with np.load(file_name) as f:
                        ouput_v1=f['ouput_v1']
                        error_v1=f['error_v1']
                  v1.append(ouput_v1)
                  error_1st_time.append(error_v1[0])
                  error_2nd_time.append(error_v1[1])
                  error_3rd_time.append(error_v1[2])
            error_list.append(error_1st_time)
            error_list.append(error_2nd_time)
            error_list.append(error_3rd_time)
            print('v1:',v1)
            print('error_list:',error_list)
            np.savez(dir+'/output_time_contrast6',v1=v1,error_list=error_list)#储存数据
            
      else:
            file_name=dir+'/Contrast10/output_vector.npz'
            with np.load(file_name) as f:
                  sampleID=f['sampleID']
                  unattacked_output_v1=f['unattacked_output_v1'] 
                  attack_v1=f['attack_v1']
                  error_vector_v1=f['error_vector_v1']
            return sampleID,unattacked_output_v1,attack_v1,error_vector_v1

#读取整合后的数据           
def read_data(dir):
      file_name=dir+'/output_time_contrast6.npz'
      with np.load(file_name) as f:
            v1=f['v1']
            error_list=f['error_list']
      return v1,error_list

def plot_time(v1,error_list,dir):
      x=[1,2,3,4,5]  #横坐标：时间
      plt.figure(figsize=(8, 8))
      ax1 = plt.subplot(111)
      for i in range(3):
            plt.plot(x, error_list[i],label="{}".format(i+1))
            ax1.legend(loc=i)
            if i==2:
                  for a, b in zip(x, np.around(error_list[i],4)):  # 添加这个循坏显示坐标
                        plt.text(a, b, b, color = "b",ha='center', va='bottom', fontsize=10)
      plt.xlim((0,5.5))
      plt.ylim((np.min(error_list)-2,np.max(error_list)+2))
      ax1.set_ylabel('error');
      ax1.set_xlabel('time/s')
      
      ax2 = ax1.twinx() 
      ax2.plot(x, v1, 'r'+'o-',label = "norm2(fr) for the 0th time")
      ax2.legend(loc=4)
      plt.ylim((np.min(v1)-2),np.max(v1)+2)
      ax2.set_ylabel('norm2(fr) for the 0th time')
      for a, b in zip(x, np.around(v1,2)):  
            plt.text(a, b, b,color = "m", ha='center', va='bottom', fontsize=10)

      #保存图片
      if not os.path.exists(dir):
                  os.mkdir(dir)
      plt.savefig(os.path.join(dir, 'repeat_time_contrast6.png'))#第一个是指存储路径，第二个是图片名字
      plt.close()
      plt.clf()

start =time.perf_counter()

if __name__ == "__main__" :
     
      delta_list=np.zeros(3)
      dir='/home/zhaobenyan/data/dw_test_new/Repeatability_time'#储存数据的路径
      if not os.path.exists(dir):
            os.mkdir(dir)
      file_name='/home/zhaobenyan/data/resource_contrast32_new/static_color-grid_6.bin'
      #Generate_data(delta_list,file_name,dir)#每改变一次时间运行一次，后面都直接读取数据即可
      #merge_data(dir)              #数据生成好之后运行一次
      v1,error_list=read_data(dir)
      plot_time(v1,error_list,dir)

end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))