from attack_dw import GaussianNoiseTestDw
import matplotlib.pyplot as plt
import numpy as np
import time
import os
np.seterr(divide='ignore',invalid='ignore')
from attack_dw import ljy_bin_image_to_tensor
np.seterr(divide='ignore',invalid='ignore')#忽略出现0/0时的警告
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

#对同一个grating重复十次，输出fr的二范数以及每一次与第0次的误差的二范数
def repeat_single(delta_list,file_name,dir):
      '''
      生成并储存数据，运行一次即可，若模型有所改动则需重新生成数据
      dalta_list:在这个实验中为全0向量
      file_name:进行测试所用grating所在路径
      dir：储存数据的路径，会生成两个文件，分别储存mode=='norm'和'vector'的数据，详见下一个函数read_data的注释
      '''
      X_grating = ljy_bin_image_to_tensor(file_name)
      if not os.path.exists(dir):
            os.mkdir(dir)
      output=GaussianNoiseTestDw(X_grating)
      output.get_noised(delta_list)
      output.get_results()
      sampleID,unattacked_output_v1, attack_v1, error_vector_v1=output.get_vector()
      np.savez(dir+'/output_vector',sampleID=sampleID,unattacked_output_v1=unattacked_output_v1, attack_v1=attack_v1, error_vector_v1=error_vector_v1)#储存数据
      ouput_v1,error_v1=output.get_norm()
      np.savez(dir+'/output_norm',ouput_v1=ouput_v1,error_v1=error_v1)#储存数据


def read_data(dir,mode='norm'):
      '''
      读取数据 默认mode=='norm'
      mode=='norm',读取攻击之后fr的二范数（delta,1)和error的二范数(delta,1)
      mode=='vector',读取神经元编号sampleID（3840,1）攻击之前的fr（1，3840）,
                  攻击之后的fr（delta，3840），攻击之后的error（delta,3840)向量
      '''
      if mode=='norm':
            file_name=dir+'/output_norm.npz'
            with np.load(file_name) as f:
                  output_v1=f['ouput_v1']
                  error_v1=f['error_v1']
            print('ouput_v1:',output_v1)
            print('error_v1:',error_v1)
            return output_v1,error_v1
      if mode=='vector':
            file_name=dir+'/output_vector.npz'
            with np.load(file_name) as f:
                  sampleID=f['sampleID']
                  unattacked_output_v1=f['unattacked_output_v1'] 
                  attack_v1=f['attack_v1']
                  error_vector_v1=f['error_vector_v1']
            return sampleID,unattacked_output_v1,attack_v1,error_vector_v1

def plot_repeat(output_v1,error_v1,dir):
      '''
      画图：横坐标为重复的次数=delta_list的长度（设置为10）
      纵坐标1为error的二范数，纵坐标2为fr的二范数
      dir为储存图片的路径，尽量与上一个函数储存输出的路径一致
      '''
      x=np.arange(10)+1  #[1,2,...,10]
      plt.figure(figsize=(8, 8))
      ax1 = plt.subplot(111)
      plt.plot(x, error_v1, 'r'+'o-',label="error_v1")
      ax1.legend(loc=1)
      plt.xlim((0,11))
      plt.ylim((np.min(error_v1)-0.5,np.max(error_v1)+0.5))
      ax1.set_ylabel('error_v1');
      ax1.set_xlabel('repeat_times')
      for a, b in zip(x, np.around(error_v1,2)):  # 添加这个循坏显示坐标
            plt.text(a, b, b, color = "b",ha='center', va='bottom', fontsize=10)
      ax2 = ax1.twinx() 
      ax2.plot(x, output_v1, 'g'+'o-',label = "v1")
      ax2.legend(loc=2)
      plt.ylim((np.min(output_v1)-0.5),np.max(output_v1)+0.5)
      ax2.set_ylabel('v1')
      for a, b in zip(x, np.around(output_v1,2)):  # 添加这个循坏显示坐标
            plt.text(a, b, b,color = "b", ha='center', va='bottom', fontsize=10)

      #保存图片
      if not os.path.exists(dir):
                  os.mkdir(dir)
      plt.savefig(os.path.join(dir, 'repeat_contrast6.png'))#第一个是指存储路径，第二个是图片名字
      plt.close()
      plt.clf()

start =time.perf_counter()
if __name__ == "__main__" :
      delta_list=np.zeros(10)
      dir='/home/zhaobenyan/data/dw_test_new/Repeatability_noseed'#储存数据的路径
      file_name='/home/zhaobenyan/data/resource_contrast32_new/static_color-grid_6.bin'
      repeat_single(delta_list,file_name,dir)#只有第一次需要运行，后面都直接读取数据即可
      output_v1,error_v1=read_data(dir)
      plot_repeat(output_v1,error_v1,dir)
end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))