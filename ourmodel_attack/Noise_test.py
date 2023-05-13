from attack_dw import GaussianNoiseTestDw
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from pandas import Series,DataFrame
# import seaborn as sns
# import palettable #python颜色库
np.seterr(divide='ignore',invalid='ignore')
from attack_dw import image_to_tensor
np.seterr(divide='ignore',invalid='ignore')#忽略出现0/0时的警告
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

'''
进行高斯噪声攻击
mode='gray' :灰度图; mode = 'grating':grating
'''

def gauss_attak(delta_list,input,dir):
      '''
      生成并储存数据，运行一次即可，若模型有所改动则需重新生成数据
      dalta_list:在这个实验中为delta_list = np.arange(0, 0.5, 0.01)
      file_name:进行测试所用grating所在路径
      dir：储存数据的路径，会生成两个文件，分别储存mode=='norm'和'vector'的数据，详见下一个函数read_data的注释
      '''
      output=GaussianNoiseTestDw(input)
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
      mode=='vector',读取神经元编号sampleID（3066,1）攻击之前的fr（1，3066）,
                  攻击之后的fr（delta，3066），攻击之后的error（delta,3066)向量
      '''
      if mode=='norm':
            file_name=dir+'/output_norm.npz'
            with np.load(file_name) as f:
                  output_v1=f['ouput_v1']
                  error_v1=f['error_v1']
            #print('ouput_v1:',output_v1)
            #print('error_v1:',error_v1)
            return output_v1,error_v1
      if mode=='vector':
            file_name=dir+'/output_vector.npz'
            with np.load(file_name) as f:
                  sampleID=f['sampleID']
                  unattacked_output_v1=f['unattacked_output_v1'] 
                  attack_v1=f['attack_v1']
                  error_vector_v1=f['error_vector_v1']
            return sampleID,unattacked_output_v1,attack_v1,error_vector_v1

def compute_mean_std(dir_output):
      sampleID,unattacked_output_v1,attack_v1,error_vector_v1=read_data(dir_output,mode='vector')
      error_mean=np.mean(np.abs(error_vector_v1),axis=1)    #fr误差均值
      fr_mean=np.mean(attack_v1,axis=1)
      num_nozero=np.count_nonzero(attack_v1==0,axis=1)
      print(num_nozero)
      return error_mean,fr_mean

def plot_gauss(output_v1,error_v1,delta_list,contrast,dir):
      '''
      画图：横坐标为重复的次数=delta_list的长度（设置为10）
      纵坐标1为error的二范数，纵坐标2为fr的二范数
      dir为储存图片的路径，尽量与上一个函数储存输出的路径一致
      '''
      x=delta_list
      plt.figure(figsize=(8, 8))
      ax1 = plt.subplot(111)
      plt.plot(x, error_v1, 'r'+'o-',label="error")
      ax1.legend(loc=2)
      plt.xlim((0,0.52))
      plt.ylim((0,np.max(error_v1)+10))
      ax1.set_ylabel('error');
      ax1.set_xlabel('delta')
      # for a, b in zip(x, np.around(error_v1,2)):  # 添加这个循坏显示坐标
      #       plt.text(a, b, b, color = "g",ha='center', va='bottom', fontsize=10)
      ax2 = ax1.twinx() 
      ax2.plot(x, output_v1, 'g'+'o-',label = "fr")
      ax2.legend(loc=1)
      plt.ylim((0,np.max(output_v1)+10))
      ax2.set_ylabel('fr')
      # for a, b in zip(x, np.around(output_v1,2)):  # 添加这个循坏显示坐标
      #       plt.text(a, b, b,color = "b", ha='center', va='bottom', fontsize=10)
      plt.title('norm2_contrast{}'.format(contrast))
      #保存图片
      if not os.path.exists(dir):
                  os.mkdir(dir)
      plt.savefig(os.path.join(dir, 'gauss_attack.png'))#第一个是指存储路径，第二个是图片名字
      plt.close()
      plt.clf()

def plot_mean(error_mean,fr_mean,delta_list,contrast,dir):
      '''
      画图：横坐标为重复的次数=delta_list的长度（设置为50）
      纵坐标1为error的均值，纵坐标2为fr的均值
      dir为储存图片的路径，尽量与上一个函数储存输出的路径一致
      '''
      x=delta_list
      plt.figure(figsize=(8, 8))
      ax1 = plt.subplot(111)
      plt.plot(x, error_mean, 'r'+'o-',label="error_mean/hz(left)")
      ax1.legend(loc=2)
      plt.xlim((0,0.52))
      plt.ylim((0,np.max(error_mean)))
      ax1.set_ylabel('error_mean/hz');
      ax1.set_xlabel('delta')
      ax2 = ax1.twinx() 
      ax2.plot(x, fr_mean, 'g'+'o-',label = "fr_mean/hz(right)")
      ax2.legend(loc=1)
      plt.ylim((0,np.max(fr_mean)+1))
      ax2.set_ylabel('fr_mean/hz')
      plt.title('mean_contrast{}'.format(contrast))

      #保存图片
      if not os.path.exists(dir):
                  os.mkdir(dir)
      plt.savefig(os.path.join(dir, 'gauss_attack_mean.png'))#第一个是指存储路径，第二个是图片名字
      plt.close()
      plt.clf()

def main(mode, delta_list, dir):
      if mode == 'gray':            
            #灰度图实验
            X = 0.5 * np.ones(shape=[32 ,32, 3])
            dir_gray=dir+'/gray'
            if not os.path.exists(dir_gray):
                  os.mkdir(dir_gray)
            gauss_attak(delta_list,X,dir_gray)#只有第一次需要运行，后面都直接读取数据即可
            output_v1,error_v1=read_data(dir_gray)
            plot_gauss(output_v1,error_v1,delta_list,dir_gray)
            
      elif mode == 'grating':
            #grating实验
            for i in range(1,11):  
                  #测试用的grating所在目录                
                  file_name='/home/zhaobenyan/dataset/grating/grating_32x32_frameRate1/static_color-grid_{}.bin'.format(i)
                  X_grating = image_to_tensor(file_name)
                  dir_grating=dir+'/Contrast{}'.format(i)
                  if not os.path.exists(dir_grating):
                        os.mkdir(dir_grating)
                  #gauss_attak(delta_list,X_grating,dir_grating)#只有第一次需要运行，后面都直接读取数据即可
                  output_v1,error_v1=read_data(dir_grating)
                  plot_gauss(output_v1,error_v1,delta_list,i,dir_grating)
                  error_mean,fr_mean=compute_mean_std(dir_grating)
                  plot_mean(error_mean,fr_mean,delta_list,i,dir_grating)
            

start =time.perf_counter()

if __name__ == "__main__" :
      mode='grating'
      delta_list = np.arange(0, 0.5, 0.01)
      dir='/home/zhaobenyan/dataset/output/attack'#储存数据的路径
      dir_output=dir+'/Contrast8'
      if not os.path.exists(dir):
            os.mkdir(dir)
      #compute_mean_std(dir_output)
      main(mode, delta_list, dir)
      # error_mean,fr_mean=compute_mean_std(dir_output)
      # plot_mean(error_mean,fr_mean,delta_list,dir_output)

end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))

