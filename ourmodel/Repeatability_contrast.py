from attack_dw import GaussianNoiseTestDw
import matplotlib.pyplot as plt
import numpy as np
import time
import os
np.seterr(divide='ignore',invalid='ignore')
from attack_dw import ljy_bin_image_to_tensor
np.seterr(divide='ignore',invalid='ignore')#忽略出现0/0时的警告
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

#对不同contrast数据重复3次实验
def Generate_data(delta_list,dir):
      '''
      生成并储存数据，运行一次即可，若模型有所改动则需重新生成数据
      dalta_list:在这个实验中为全0向量
      dir：储存数据的路径，不同contrast会生成两个文件，分别储存mode=='norm'和'vector'的数据，详见下一个函数read_data的注释
      '''
      for i in range(1,11):  
            #测试用的grating所在目录                
            file_name='/home/zhaobenyan/data/resource_contrast32_new/static_color-grid_{}.bin'.format(i)
            dir_grating=dir+'/Contrast{}'.format(i)
            if not os.path.exists(dir_grating):
                  os.mkdir(dir_grating)
            X_grating = ljy_bin_image_to_tensor(file_name)
            output=GaussianNoiseTestDw(X_grating)
            output.get_noised(delta_list)
            output.get_results()
            sampleID,unattacked_output_v1, attack_v1, error_vector_v1=output.get_vector()
            np.savez(dir_grating+'/output_vector',sampleID=sampleID,unattacked_output_v1=unattacked_output_v1, attack_v1=attack_v1, error_vector_v1=error_vector_v1)#储存数据
            v1_before,error_v1=output.get_norm_before()
            np.savez(dir_grating+'/output_norm_before',v1_before=v1_before,error_v1=error_v1)#储存数据


def merge_data(dir,mode='norm'):
      '''
      整合数据，运行一次即可， 默认mode=='norm'
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
            for i in range(1,11):
                  file_name=dir+'/Contrast{}/output_norm_before.npz'.format(i)
                  with np.load(file_name) as f:
                        v1_before=f['v1_before']
                        error_v1=f['error_v1']
                        print('v1_before:',v1_before)
                  v1.append(v1_before)
                  print(v1)
                  error_1st_time.append(error_v1[0])
                  error_2nd_time.append(error_v1[1])
                  error_3rd_time.append(error_v1[2])
            error_1st_time=list(np.nan_to_num(np.array(error_1st_time)/np.array(v1)))
            error_2nd_time=list(np.nan_to_num(np.array(error_2nd_time)/np.array(v1)))
            error_3rd_time=list(np.nan_to_num(np.array(error_3rd_time)/np.array(v1)))
            error_list.append(error_1st_time)
            error_list.append(error_2nd_time)
            error_list.append(error_3rd_time)
            print('v1:',v1)
            print('error_list:',error_list)
            np.savez(dir+'/output_plot',v1=v1,error_list=error_list)#储存数据
            
      else:
            file_name=dir+'/Contrast{}/output_vector.npz'
            with np.load(file_name) as f:
                  sampleID=f['sampleID']
                  unattacked_output_v1=f['unattacked_output_v1'] 
                  attack_v1=f['attack_v1']
                  error_vector_v1=f['error_vector_v1']
            return sampleID,unattacked_output_v1,attack_v1,error_vector_v1

#读取整合后的数据           
def read_data(dir):
      file_name=dir+'/output_plot.npz'
      with np.load(file_name) as f:
            v1=f['v1']
            error_list=f['error_list']
      return v1,error_list

def plot_contrast(v1,error_list,dir):
      x=np.arange(0.05,0.55,0.05)
      plt.figure(figsize=(8, 8))
      ax1 = plt.subplot(111)
      for i in range(3):
            plt.plot(x, error_list[i],label="{}".format(i+1))
            ax1.legend(loc=i)
            if i==2:
                  for a, b in zip(x, np.around(error_list[i],4)):  # 添加这个循坏显示坐标
                        plt.text(a, b, b, color = "b",ha='center', va='bottom', fontsize=10)
      plt.xlim((0,0.6))
      plt.ylim((np.min(error_list)-0.02,np.max(error_list)+0.02))
      ax1.set_ylabel('error_v1');
      ax1.set_xlabel('contrast')
      
      ax2 = ax1.twinx() 
      ax2.plot(x, v1, 'r'+'o-',label = "v1")
      ax2.legend(loc=4)
      plt.ylim((np.min(v1)-10),np.max(v1)+10)
      ax2.set_ylabel('v1')
      for a, b in zip(x, np.around(v1,2)):  
            plt.text(a, b, b,color = "m", ha='center', va='bottom', fontsize=10)

      #保存图片
      if not os.path.exists(dir):
                  os.mkdir(dir)
      plt.savefig(os.path.join(dir, 'repeat_contrast.png'))#第一个是指存储路径，第二个是图片名字
      plt.close()
      plt.clf()

start =time.perf_counter()
if __name__ == "__main__" :
      delta_list=np.zeros(3)
      dir='/home/zhaobenyan/data/dw_test_new/Repeatability_contrast_noseed_noori'#储存数据的路径
      if not os.path.exists(dir):
            os.mkdir(dir)
      Generate_data(delta_list,dir)#只有第一次需要运行，后面都直接读取数据即可
      merge_data(dir)  #只有第一次需要运行
      v1,error_list=read_data(dir)
      print('v1.shape:',len(v1))
      plot_contrast(v1,error_list,dir)
end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))