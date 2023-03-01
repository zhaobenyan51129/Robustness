import torch
import time
import numpy as np
import os
from attack_vit import GaussianNoiseTestVit, Plot

'''
进行高斯噪声攻击
mode='gray' :灰度图; mode = 'grating':grating
'''

def gauss_attak(input,dir,delta_list,encoder_num):
      '''
      生成并储存数据，运行一次即可，若模型有所改动则需重新生成数据
      input:输入的图片  tensor[224,224,3]
      dalta_list:在这个实验中为np.arange(0, 0.5, 0.01)
      dir:储存数据的路径
      encoder_num:encoder的层数(最多12层)
      '''
      output=GaussianNoiseTestVit(input,encoder_num)
      output.get_noised(delta_list)
      output.get_results()
      embedding_list, encoder_list, logit_list=output.get_all()
      np.savez(dir+'/output_list',embedding_list=embedding_list,encoder_list=encoder_list,logit_list=logit_list)#储存数据
      

def read_data(dir):
      '''读取数据,需要哪个读哪个'''
      file_name=dir+'/output_list.npz'
      with np.load(file_name,allow_pickle=True) as f:
            embedding_list=f['embedding_list']
            encoder_list=f['encoder_list']
            logit_list=f['logit_list']
      return embedding_list, encoder_list, logit_list

#载入多图攻击的100张图
def read_image():
      batch_file = '/home/zhaobenyan/robustness/datas/images_labels_224.pth'
      images_labels = torch.load(batch_file)
      images = images_labels['images']  #(torch.Size([100, 3, 224, 224])
      labels = images_labels['labels']  #torch.Size([100]))
      imgs=images.numpy().transpose(0,2,3,1)  #(100, 224, 224, 3)
      return imgs

#画图
def plot_gauss(embedding_list, encoder_list, logit_list, encoder_num, dir, delta_list):
      plot=Plot(dir,delta_list)
      plot.plot_embedding(embedding_list)
      plot.plot_encoder(encoder_num, encoder_list)
      plot.plot_logit(logit_list)

def main(mode, delta_list, dir,encoder_num):
      if mode == 'gray':            
            #灰度图实验
            X = 0.5 * np.ones(shape=[224, 224, 3])
            dir_gray=dir+'/gray'
            print('dir_gray:',type(dir_gray))
            if not os.path.exists(dir_gray):
                  os.mkdir(dir_gray)
            #gauss_attak(X,dir_gray,delta_list,encoder_num)#只有第一次需要运行，后面都直接读取数据即可
            embedding_list, encoder_list, logit_list=read_data(dir_gray)
            plot_gauss(embedding_list, encoder_list, logit_list, encoder_num, dir_gray, delta_list)
            
      elif mode == 'grating':
            #grating实验
            imgs=read_image()
            for i in range(3):
                  img=imgs[i,:,:,:]
                  dir_img=dir+'/image{}'.format(i+1)
                  if not os.path.exists(dir_img):
                        os.mkdir(dir_img)
                  gauss_attak(img,dir_img,delta_list,encoder_num)#只有第一次需要运行，后面都直接读取数据即可
                  embedding_list, encoder_list, logit_list=read_data(dir_img)
                  plot_gauss(embedding_list, encoder_list, logit_list,encoder_num,dir_img,delta_list)
           

start =time.perf_counter()

if __name__ == "__main__" :
      mode='grating'
      delta_list = np.arange(0, 0.5, 0.01)
      dir='/home/zhaobenyan/data/dt_vit_try'#储存数据的路径
      encoder_num=3
      if not os.path.exists(dir):
            os.mkdir(dir)
      main(mode, delta_list, dir, encoder_num)

end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))

