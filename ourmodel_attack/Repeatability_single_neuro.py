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
      这里与Repeatability_contrast.py生成数据相同，具体可参见Repeatability_contrast.py对应部分
      如果已经生成过，可以直接读取output_vector.npz中的数据
      dalta_list:在这个实验中为全0向量
      dir：储存数据的路径，不同contrast会生成两个文件
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
            ouput_v1,error_v1=output.get_norm()
            np.savez(dir_grating+'/output_norm',ouput_v1=ouput_v1,error_v1=error_v1)#储存数据


def read_data(file_name):
      '''
      sampleID (3084,)
      unattacked_output_v1 (3084,)
      attack_v1 (3, 3084)
      error_vector_v1 (3, 3084)
      '''
      with np.load(file_name) as f:
            sampleID=f['sampleID']
            unattacked_output_v1=f['unattacked_output_v1'] 
            attack_v1=f['attack_v1']
            error_vector_v1=f['error_vector_v1']
      return sampleID, unattacked_output_v1, attack_v1,error_vector_v1   


def plot_single_neuro(sampleID,unattacked_output_v1,attack_v1,error_vector_v1,dir):
      num=3084  #设置画前num个neuro，3084就是全都画
      unattacked_output_v1=unattacked_output_v1[0:num]
      attack_v1=attack_v1[:,0:num]
      error_vector_v1=error_vector_v1[:,0:num]
      x=sampleID[0:num]

      plt.figure(figsize=(50, 5))
      ax1 = plt.subplot(111)
      for i in range(3):
            plt.plot(x, error_vector_v1[i,:],label="{}".format(i+1))
            ax1.legend(loc=i)
      plt.xlim((0,np.max(x)+2))
      plt.ylim((np.min(error_vector_v1)-2,np.max(error_vector_v1)+2))
      ax1.set_ylabel('v1');
      ax1.set_xlabel('neuro')
      # for a, b in zip(x, np.around(self.error_v1,2)):  # 添加这个循坏显示坐标
      #     plt.text(a, b, b, color = "b",ha='center', va='bottom', fontsize=10)
      ax2 = ax1.twinx() 
      ax2.plot(x, unattacked_output_v1, 'r'+'o-',label = "0")
      ax2.legend(loc=4)
      plt.ylim((np.min(unattacked_output_v1)-10),np.max(unattacked_output_v1)+10)
      ax2.set_ylabel('original_output')
      # for a, b in zip(x, np.around(self.v1,2)):  # 添加这个循坏显示坐标
      #     plt.text(a, b, b,color = "b", ha='center', va='bottom', fontsize=10)

      #保存图片
      if not os.path.exists(dir):
                  os.mkdir(dir)
      plt.savefig(os.path.join(dir, 'repeat_single_neuro.png'))#第一个是指存储路径，第二个是图片名字
      plt.close()
      plt.clf()

start =time.perf_counter()
if __name__ == "__main__" :
      delta_list=np.zeros(3)
      dir='/home/zhaobenyan/data/dw_test_new/Repeatability_contrast_noseed'#储存数据的路径,
      if not os.path.exists(dir):
            os.mkdir(dir)
      #Generate_data(delta_list,dir)#只有第一次需要运行，后面都直接读取数据即可
      for i in range(1,11):
            #这里和Repeatability_contrast可以共用数据
            file_name=dir+'/Contrast{}/output_vector.npz'.format(i)
            sampleID, unattacked_output_v1, attack_v1, error_vector_v1=read_data(file_name)
            plot_single_neuro(sampleID, unattacked_output_v1, attack_v1, error_vector_v1,dir+'/Contrast{}'.format(i))     

end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))