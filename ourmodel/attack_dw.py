import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import copy 
import glob
from ourmodel_attack import ourmodel 

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

# 获取基准 【注意】chosen需和daiwei_pack中保持一致
def get_standard(chosen=False):

    with open('/home/zhaobenyan/data/benchmark_static_32pixel.bin') as f:
        sampleSize = np.fromfile(f,'u4',1)[0]
        sampleID = np.fromfile(f,'u4',sampleSize)
        sample_nLGN_V1 = np.fromfile(f,'u4',sampleSize) 
        fr_max_v1 = np.fromfile(f,'f4',sampleSize)       
        fr_input_lgn = np.fromfile(f, 'f4', 3840)
        zero_middle = np.nonzero(fr_input_lgn != 0)[0]
        if chosen:#删除lgn没有响应的
            sampleID =sampleID[zero_middle]
            fr_max_v1 = fr_max_v1[zero_middle]
            fr_input_lgn = fr_input_lgn[zero_middle]
        else:
            sampleID =sampleID
            fr_max_v1 = fr_max_v1
            fr_input_lgn = fr_input_lgn

    return sampleID,zero_middle,fr_input_lgn, fr_max_v1

def daiwei_pack(data, mode='nonchosen'): # [16, 16 , 3] # mode: chosen/nonchosen
    _,zero_middle,fr_input_max, fr_max_v1 = get_standard()
    output = ourmodel(data)
    if mode == 'chosen':
        return output[0][zero_middle], output[1][zero_middle]
    else:
        return output[0], output[1]

class GaussianNoiseTestDw:

    def __init__(self, input_fig): # input: [h, w, 3], better numpy
        
        self.input_fig = input_fig[np.newaxis, :]  #[1,224,224,3]
        self.img_size = self.input_fig.shape[-2]
        self.model = daiwei_pack # 输入 [h, w, 3]， 输出：（中间层向量[num_neurons,]，输出层向量[num_neurons,]）
        sampleID,_,fr_max_lgn, fr_max_v1 = get_standard()
        self.sampleID=sampleID
        self.max_fr_middle = fr_max_lgn 
        self.max_fr_final = fr_max_v1 
        self.plot_color = 'r'
        
        self.unattacked_output = self.model(self.input_fig) # [k_medium,], [k_final,]
        #print('初始output大小', self.unattacked_output[0].shape, self.unattacked_output[1].shape)
        #[3066,],[3066,]
    
    def get_noised(self, delta_list, grating=False, fix=True,clipping=False): 
        '''
        生成白噪声进行攻击
        grating设置为TRUE的时候只攻击前两个channel
        fix:固定白噪声的pattern
            fix=True:先生成(0,1)正态分布standard_preturb,然后对每个delta,
            用delta乘这个standard_preturb作为噪声（每个噪声值相差一个倍数）
            fix=False:每个delta都随机生成一个（0，delta）的噪声
        clipping:是否进行clip操作
        '''
        self.delta_list = delta_list
        noise=torch.load('/home/zhaobenyan/robustness/datas/normal_noise_{}.pth'.format(self.img_size)) #噪声
        
        # 生成被白噪声攻击过的图片list
        if fix:
            standard_preturb = noise[np.newaxis, :]  #[224,224,3]->[1,224,224,3]
            if grating:
                standard_preturb[:, :, :, -1] = 0.0
            self.preturb = [delta * standard_preturb for delta in delta_list]
            self.preturb = np.concatenate(self.preturb, axis=0)
        else:
            self.preturb = [np.random.normal(loc=0.0, scale= delta, size=[1, self.img_size, self.img_size, 3]) for delta in delta_list]
            self.preturb = np.concatenate(self.preturb, axis=0)
            if grating:
                self.preturb[:, :, :, -1] = 0.0

        if clipping:
            self.noised_inputs = np.clip(self.input_fig + self.preturb, 0, 1) #[delta,224,224,3]
        else:
            self.noised_inputs = self.input_fig + self.preturb  #[delta,224,224,3]

        #print('preturb shape:',self.preturb.shape)  #[delta,224,224,3]
        self.preturb_norm = np.linalg.norm(self.preturb.reshape([len(self.preturb), -1]), axis=-1)
        
        #统计被clip的数量
        self.images_without_clip=self.input_fig + self.preturb #[delta,224,224,3]
        self.smaller_than_zero=(self.images_without_clip<0)+0 #[delta,224,224,3] #+0将（TRUE，FALSE变为1,0）
        self.smaller_than_zero_vector=self.smaller_than_zero.reshape([len(self.smaller_than_zero), -1])
        #print(self.smaller_than_zero_vector.shape) #[delta,150528]
        self.bigger_than_one=(self.images_without_clip>1)+0 
        self.bigger_than_one_vector=self.bigger_than_one.reshape([len(self.bigger_than_one), -1])
        self.clip_number=np.sum(self.smaller_than_zero_vector,axis=1)+np.sum(self.bigger_than_one_vector,axis=1)
        self.clip_ratio=self.clip_number / (3*self.img_size*self.img_size)

    def get_results(self,Norm=None):
        '''
        数据处理，得到需要的指标
        Norm:用什么范数，默认为None(二范数)
        1:1范数，2：2范数，np.inf：无穷范数
        '''
        self.Norm=Norm
        #没有攻击时的输出
        self.unattacked_output_lgn=self.unattacked_output[0]
        self.unattacked_output_v1=self.unattacked_output[1]
        #print('self.unattacked_output_v1.shape:',self.unattacked_output_v1.shape)
        # 进行攻击，所有数据维度均为[delta,3066]
        result_list = [(self.model(noised_input[None, :])) for noised_input in self.noised_inputs]
        self.attack_lgn = np.concatenate([result[0][None, :] for result in result_list], axis=0)#扰动后LGN输出
        self.attack_v1 = np.concatenate([result[1][None, :] for result in result_list], axis=0) #扰动后V1输出
        #print('self.attack_v1.shape:',self.attack_v1.shape)

        self.error_vector_lgn = self.attack_lgn - self.unattacked_output[0][None, :]   #LGN扰动前减扰动后
        self.error_vector_v1 = self.attack_v1 - self.unattacked_output[1][None, :]     #V1扰动前减扰动后

        self.output_lgn=np.linalg.norm(self.unattacked_output[0][None, :],ord=self.Norm, axis=1) #未攻击的LGN输出的二范数
        self.output_v1 = np.linalg.norm(self.unattacked_output[1][None, :], ord=self.Norm, axis=1) #未攻击的V1输出的二范数

        self.error_lgn=np.linalg.norm(self.error_vector_lgn, ord=self.Norm, axis=1)  #LGN误差二范数 误差：扰动后输出减扰动前输出
        self.error_v1=np.linalg.norm(self.error_vector_v1, ord=self.Norm, axis=1)    #V1误差二范数

        self.error_scale_lgn= np.linalg.norm(self.error_vector_lgn / self.max_fr_middle,ord=self.Norm, axis=1) #LGN误差除以max_fr的二范数
        self.error_scale_v1= np.linalg.norm(self.error_vector_v1 / self.max_fr_final,ord=self.Norm, axis=1) #V1误差除以max_fr的二范数

        self.relative_lgn = self.error_lgn / self.output_lgn        #误差的二范数 除以 攻击之前的输出的二范数
        self.relative_v1 = self.error_v1 / self.output_v1

        self.relative_scale_lgn = self.error_scale_lgn / np.linalg.norm(self.unattacked_output[0][None, :] / self.max_fr_middle,ord=self.Norm, axis=1) 
        self.relative_scale_v1 = self.error_scale_v1 / np.linalg.norm(self.unattacked_output[1][None, :] / self.max_fr_final,ord=self.Norm, axis=1) 
        #上述的scale

        #横坐标为norm(clip(x+delta)-x)
        self.clip_x=(self.noised_inputs-self.input_fig).reshape([len(self.noised_inputs), -1])
        self.clip_x_norm=np.linalg.norm(self.clip_x,ord=self.Norm, axis=1)

        self.v1=np.linalg.norm(self.attack_v1,ord=self.Norm, axis=1)  #攻击后v1输出的二范数 [delta,]
        self.v1_before=np.linalg.norm(self.unattacked_output_v1,ord=self.Norm, axis=0) #攻击前v1输出的二范数 [1,]

    #返回神经元编号sampleID（3066,1）攻击之前的fr（1，3066）,攻击之后的fr（delta，3066），攻击之后的error（delta,3066)向量
    def get_vector(self):#,dir=None):

        return self.sampleID,self.unattacked_output_v1, self.attack_v1, self.error_vector_v1

    #返回攻击之后fr的二范数（delta,1)和error的二范数(delta,1)
    def get_norm(self):

        return self.v1, self.error_v1
    
    #返回攻击之前的fr的二范数(1,),和error的二范数(delta,1)
    def get_norm_before(self):

        return self.v1_before, self.error_v1

def show_evolution(self): # dir: 'show_evolution/子文件夹名/'
      if not os.path.exists(self.dir):
                  os.mkdir(self.dir)
      for k in range(len(self.preturb)):
                  fig, ax = plt.subplots(1, 3)
                  fig.set_size_inches(9, 3)
                  ax[0].imshow(self.input_fig[0])# [h, w, 3]
                  ax[0].set_axis_off()
                  ax[1].imshow(0.5 + self.preturb[k])
                  ax[1].set_axis_off()
                  ax[2].imshow(self.noised_inputs[k])
                  ax[2].set_axis_off()
                  plt.savefig(self.dir + f'{k}.jpg')
                  plt.close()



class Plot():
        def __init__(self,dir,delta_list):
            self.dir=dir #储存路径
            self.delta_list=delta_list #delta取值       
            
        def plot_v1(self,v1,error_v1):  #绘图V1

            self.v1=v1   #v1输出
            self.error_v1=error_v1  #v1 error

            plt.figure(figsize=(8, 8))
            ax1 = plt.subplot(111)
            plt.plot(self.delta_list, self.error_v1, 'r'+'o-',label="error_v1")
            ax1.legend(loc=1)
            plt.xlim((0,0.52))
            plt.ylim((0,np.max(self.error_v1)+2))
            ax1.set_ylabel('error_v1');
            ax1.set_xlabel('delta')
            ax2 = ax1.twinx() 
            ax2.plot(self.delta_list, self.v1, 'g'+'-',label = "v1")
            ax2.legend(loc=2)
            plt.ylim((0,np.max(self.v1)+5))
            ax2.set_ylabel('v1');

            #保存图片
            if not os.path.exists(self.dir):
                    os.mkdir(self.dir)
            plt.savefig(os.path.join(self.dir, 'V1.png'))#第一个是指存储路径，第二个是图片名字
            plt.close()
            plt.clf()

       
        def plot_contrast(self,v1,error_list):
            contrast=np.arange(0.05,0.55,0.05)
            #time=[1,2,3,4,5]
            x=contrast
            self.v1=v1
            self.error_list=error_list
            plt.figure(figsize=(8, 8))
            ax1 = plt.subplot(111)
            for i in range(3):
                plt.plot(x, self.error_list[i],label="{}".format(i+1))
                ax1.legend(loc=i)
                if i==2:
                    for a, b in zip(x, np.around(self.error_list[i],4)):  # 添加这个循坏显示坐标
                        plt.text(a, b, b, color = "b",ha='center', va='bottom', fontsize=10)
            plt.xlim((0,0.6))
            plt.ylim((np.min(self.error_list)-0.02,np.max(self.error_list)+0.02))
            ax1.set_ylabel('error_v1');
            ax1.set_xlabel('contrast')
            
            ax2 = ax1.twinx() 
            ax2.plot(x, self.v1, 'r'+'o-',label = "v1")
            ax2.legend(loc=4)
            plt.ylim((np.min(self.v1)-10),np.max(self.v1)+10)
            ax2.set_ylabel('v1')
            for a, b in zip(x, np.around(self.v1,2)):  
                plt.text(a, b, b,color = "m", ha='center', va='bottom', fontsize=10)

            #保存图片
            if not os.path.exists(self.dir):
                    os.mkdir(self.dir)
            plt.savefig(os.path.join(self.dir, 'repeat_contrast_new.png'))#第一个是指存储路径，第二个是图片名字
            plt.close()
            plt.clf()

            
#运行上述类的函数
def experiment(input,dir,delta_list):  #input:[32,32,3]  dir:生成图片的储存路径
    output=GaussianNoiseTestDw(input)
    output.get_noised(delta_list)
    v1,error_v1=output.get_results()
    plot=Plot(dir,delta_list)
    plot.plot_v1(v1,error_v1)
    np.savez(dir+'/output',v1=v1,error_v1=error_v1)



#读取输出
def read_output(file_name):
      with np.load(file_name) as f:
            v1=f['v1']
            error_v1=f['error_v1']
      return v1, error_v1

