from generate import *
import numpy as np
from UMRtools import *
import matplotlib.pyplot as plt
from PIL import Image
import time
import pandas as pd
import seaborn as sns
import os.path
import shutil
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示
import warnings
warnings.filterwarnings('ignore')

#重新配置ourmodel,要建立在已经配置过的基础上，否则要从修改路径开始
def Reconfigure():
   os.system(f'cd {SRC_DIR}/ && ./compile-nonhpc') #编译patchfast 生成模拟v1的程序 
   os.system(f'cd {SRC_DIR}/ && ./gompile-nonhpc') #编译gencon 生成v1里的连接矩阵
   os.system(f'cd {SRC_DIR}/ && ./rompile-nonhpc') #编译retino 生成lgn->v1的连接矩阵
   os.system(f'cd {PATCHV1_DIR}/ && python {PATCHV1_DIR}pinwheel_disk.py') #设置模型中具体神经元的视野范围等
   os.system(f'cd {SET_UP_DIR} && {BIN_DIR}retino -c {SRC_DIR}test_lgn.cfg')
   os.system(f'cd {SET_UP_DIR} && {BIN_DIR}genCon -c {SRC_DIR}test_v1.cfg')


# 更改指定minimal.cfg文件的值 
def Modify_minimalcfg(minimal_cfg_file_path,plotdw,*args):

   Simulation_duration,fStimulus,frameRate,output_suffix = args
  
   with open(minimal_cfg_file_path, 'r') as file:
   # 读取文件的每一行，并用列表保存
      lines = file.readlines()
      for i in range(len(lines)):
         if "nt =" in lines[i] and "#nt =" not in lines[i] and lines[i][0]=='n':
               lines[i] = f"nt = {int(8000*Simulation_duration)} \n"
               if log_on:
                  print(f"changed nt = {int(8000*Simulation_duration)}") 
         if "fStimulus =" in lines[i] and "#fStimulus =" not in lines[i]:
               lines[i] = f"fStimulus = {fStimulus} \n"
               if log_on:
                  print(f"changed fStimulus = {fStimulus}") 
         if "frameRate =" in lines[i] and "#frameRate =" not in lines[i]:
               lines[i] = f"frameRate = {frameRate} \n"
               if log_on:
                  print(f"changed frameRate = {frameRate}") 
         if "output_suffix =" in lines[i] and "#output_suffix =" not in lines[i]:
               lines[i] = f"output_suffix = {output_suffix} \n"
               if log_on:
                  print(f"changed output_suffix = {output_suffix}") 
         if plotdw==True:
            if "minimal =" in lines[i] and "#minimal =" not in lines[i]:
               lines[i] = f"minimal = False \n"
               if log_on:
                  print(f"changed minimal = False") 
         else:
             if "minimal =" in lines[i] and "#minimal =" not in lines[i]:
               lines[i] = f"minimal = True \n"
               if log_on:
                  print(f"changed minimal = True") 
      with open(minimal_cfg_file_path, 'w') as file:
         # 将修改后的行重新写入到文件中
         file.writelines(lines)

def Modify_Parameters(frameRate, Simulation_duration, C, P, SF, D, size, stimulus_name, log_on = True, singleOri = True, plotdw=False, plotTC=False):
   '''
   frameRate:帧率（一秒几帧，=1就是静止的,>1就是动的)
   Simulation_duration:模拟时长(s)
   C:contrast,范围在[0,0.5]
   P:#相位,范围[0,2pi)
   SF:spatial frequency,建议范围在10到40之间
   D:方向,范围[0,pi]
   size: :图片大小
   stimulus_name：stimulus的名字（即调用generate.py生成文件时的文件名） 
   log_on:是否打印一些中间值,默认为True
   singleOri:等于True的话就一次只输入一张图片,默认为True
   plotdw:是否调用戴老师的画图函数（会生成很多图像，运行比较慢，默认不画）
      注意，plotdw=True的时候不会生成spikeCount-...bin文件
   plotTC:是否画出tuning curve,默认为False
      注意：plotdw=True时plotTC才会发挥作用（具体见ori.sh)
   cfg_file_id: 指定使用的minimal配置文件编号,默认为空（即使用minimal.cfg作为cfg文件）
   '''
   generate_input(gratings_path,C,P,SF,D,frameRate,size,stimulus_name)  
   bin_files = sorted([filename for filename in os.listdir(gratings_path) if filename.endswith('.bin') and not filename.endswith('_cfg.bin') and filename.startswith(stimulus_name)], key=lambda x: int(x.split('_')[-1].split('.')[0]))

   for file in bin_files:
      shutil.copy(os.path.join(gratings_path, file), RESOURCE_DIR)

   if log_on:
      print(f"bin_files={bin_files}")
      print(f"{len(bin_files)} files have been copied to {RESOURCE_DIR}")

   if singleOri:
      cfg_file_id=''
      output_suffix = f'merge_test_0'
      fStimulus=bin_files[0]
      fStimulus_path=gratings_path+fStimulus
      minimal_cfg_file_path = f"{SRC_DIR}minimal{cfg_file_id}.cfg"
      Modify_minimalcfg(minimal_cfg_file_path,plotdw,Simulation_duration,fStimulus,frameRate,output_suffix)
      if log_on:
         print(f"fStimulus={RESOURCE_DIR}{fStimulus}")
         x_1,x_2,x_3,x_4,x_5,x_6=read_grating_bin(fStimulus_path)
         nFrame=x_2[0]
         print(f"nFrame={nFrame}")
         file_name = os.path.splitext(fStimulus)[0]
         cfg_file_path = os.path.join(gratings_path, file_name + "_cfg.bin")
         sf, ori, phase, contrast=read_grating_cfg(cfg_file_path)
         print(f"sf={sf},ori={ori},phase={phase},contrast={contrast}")
   else:
      cfg_file_ids = [i for i in range(1, 7)]
      for cfg_file_id in cfg_file_ids:
         fStimulus=bin_files[cfg_file_id-1]
         fStimulus_path=gratings_path+fStimulus
         output_suffix = f'merge_test_{cfg_file_id}'
         minimal_cfg_file_path = f"{SRC_DIR}minimal{cfg_file_id}.cfg"
         Modify_minimalcfg(minimal_cfg_file_path, plotdw, Simulation_duration, fStimulus, frameRate, output_suffix)
         if log_on:
            print(f"fStimulus={RESOURCE_DIR}{fStimulus}")
            x_1,x_2,x_3,x_4,x_5,x_6=read_grating_bin(fStimulus_path)
            nFrame=x_2[0]
            print(f"nFrame={nFrame}")
            file_name = os.path.splitext(fStimulus)[0]
            cfg_file_path = os.path.join(gratings_path, file_name + "_cfg.bin")
            sf, ori, phase, contrast=read_grating_cfg(cfg_file_path)
            print(f"sf={sf},ori={ori},phase={phase},contrast={contrast}")

   # 获取使用的minimalTC的路径
   minimalTC_file_path = f"{SRC_DIR}minimalTC"
   print(f"minimalTC_file_path={minimalTC_file_path}")
   if log_on:
      print(f"changed minimalTC_file = {minimalTC_file_path}")  

   # 更改minimalTC文件的值   
   with open(minimalTC_file_path, 'r') as file:
      lines = file.readlines()
      for i in range(len(lines)):
         if "singleOri=" in lines[i] and "#singleOri=" not in lines[i]:
               lines[i] = f"singleOri={singleOri} \n"
               if log_on:
                  print(f"changed singleOri = {singleOri}") 
         if "plotdw=" in lines[i] and "#plotdw =" not in lines[i]:
               lines[i] = f"plotdw={plotdw} \n"
               if log_on:
                  print(f"changed plotdw = {plotdw}") 
         if "plotTC=" in lines[i] and "#plotTC =" not in lines[i]:
               lines[i] = f"plotTC={plotTC} \n"
               if log_on:
                  print(f"changed plotTC = {plotTC}") 
         if "SF=" in lines[i] and "#SF=" not in lines[i]:
               lines[i] = f"SF={SF[0]} \n"
               if log_on:
                  print(f"changed SF={SF[0]}") 
         if singleOri==True:
            if "one_ori=" in lines[i] and "one_ori=$" not in lines[i]:
               lines[i] = f"one_ori=0 \n"
               if log_on:
                  print(f"changed one_ori = 0")
      with open(f"{SRC_DIR}/minimalTC", 'w') as file:
         file.writelines(lines)

def Run_Our_Model():
   singleOri=False
   frameRate=1
   Simulation_duration=1
   C=np.array([0.05,0.15,0.25,0.3,0.35,0.45])
   P=np.array([0.5*np.pi])
   SF = np.arange(36,40,10)#36
   D = np.array([np.pi/6])  # 方向范围[0,pi]
   size=128
   stimulus_name=f"try_contrast"
   Modify_Parameters(frameRate, Simulation_duration, C, P, SF, D, size, stimulus_name, log_on=True, singleOri=False, plotdw=True,plotTC=False)
  
   #运行代码
   start_time = time.time()
   os.system(f'cd {SRC_DIR} && ./minimalTC')
   end_time = time.time()
   total_time = round((end_time - start_time),3)
   if singleOri:
      output_suffix = f'merge_test_0'
   else:
      output_suffix = f'merge_test_1'

   LGN_spike_time,fr=read_spike(f"{PATCHFAST_DIR}sample_spikeCount-{output_suffix}.bin")
   LGN_spike_time = np.floor(LGN_spike_time).astype(int)
   
   return LGN_spike_time,fr,total_time

if __name__ == "__main__" :
   LGN_spike_time,fr,total_time=Run_Our_Model()
   print(f"LGN_spike_time={LGN_spike_time.shape}")
   print(f"v1_fr={fr.shape}")
   print(f"total_time={total_time}")