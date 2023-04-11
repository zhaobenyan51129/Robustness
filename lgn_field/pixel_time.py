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

#执行ourmodel的函数
#生成grating
def Generate_Gratings(C,P,SF,D,frameRate,size,stimulus_name):
    '''
    C:contrast,范围在[0,0.5]
    P:#相位,范围[0,2pi)
    SF:spatial frequency,建议范围在10到40之间
    D:方向,范围[0,pi]
    frameRate:帧率（一秒几帧，=1就是静止的,>1就是动的)
    size:图片大小
    stimulus_name:生成文件的文件名
    '''
    mk_path(gratings_path)
    generate_input(gratings_path,C,P,SF,D,frameRate,size,stimulus_name)
    #获取fStimulus及其.cfg文件对应的文件名
    fStimulus=Get_fStimulus_Name(gratings_path,stimulus_name)
    return fStimulus

#运行一次代码并记录时间
def Run_Our_Model(frameRate, Simulation_duration,cfg_file_id, *args):
    '''
    可变长参数为：C, P, SF, D, size,stimulus_name
    其中stimulus_name：stimulus的名字（即调用generate.py生成文件时的文件名） 
    如果可变长参数长度为0:固定fStimulus,如果不固定，每次都会生成grating并修改fStimulus,固定的话手动输入固定为哪个
    frameRate=1:静态图片
    Simulation_duration:模拟时长(s)
    log_on:是否打印一些中间值
    cfg_file_id: 指定使用的minimal配置文件编号,默认为空（即使用minimal.cfg作为cfg文件）
    repeat: 每组参数重复实验次数  
     
    '''
    log_on=True
    if len(args) == 0:
        fStimulus='gray1.bin'
    else:
        C, P, SF, D, size, stimulus_name = args
        generate_input(gratings_path,C,P,SF,D,frameRate,size,stimulus_name)
        # fStimulus=Generate_Gratings(C,P,SF,D,frameRate,size,stimulus_name)[0]
        fStimulus=stimulus_name+'.bin'
        fStimulus_path = gratings_path+fStimulus
        shutil.copy(fStimulus_path, RESOURCE_DIR)
    
    if log_on:
        print(f"fStimulus={RESOURCE_DIR}{fStimulus}")
        x_1,x_2,x_3,x_4,x_5,x_6=read_grating_bin(fStimulus_path)
        nFrame=x_2[0]
        print(f"nFrame={nFrame}")
        file_name = os.path.splitext(fStimulus)[0]
        cfg_file_path = os.path.join(gratings_path, file_name + "_cfg.bin")
        sf, ori, phase, contrast=read_grating_cfg(cfg_file_path)
        print(f"sf={sf},ori={ori},phase={phase},contrast={contrast}")
    
    # 获取使用的配置文件路径
    minimal_cfg_file_path = f"{SRC_DIR}minimal{cfg_file_id}.cfg"
    print(f"minimal_cfg_file_path={minimal_cfg_file_path}")
    if cfg_file_id=='':
        output_suffix = f'merge_test_0'
    else:
        output_suffix = f'merge_test_{cfg_file_id}'
    if log_on:
        print(f"changed minimal_cfg_file = {minimal_cfg_file_path}")  

    #更改cfg文件的值   
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
        with open(f"{SRC_DIR}/minimal{cfg_file_id}.cfg", 'w') as file:
            # 将修改后的行重新写入到文件中
            file.writelines(lines)

    #记录运行时间
    start_time = time.time()
    os.system(f'cd {PATCHFAST_DIR} && {BIN_DIR}patch_fast -c {SRC_DIR}minimal{cfg_file_id}.cfg > {model_log_path}{now_time_str()}.log')
    
    LGN_spike_time,fr=read_spike(f"{PATCHFAST_DIR}sample_spikeCount-{output_suffix}.bin")
    LGN_spike_time = np.floor(LGN_spike_time).astype(int)
    end_time = time.time()
    total_time = round((end_time - start_time),3)
    
    return LGN_spike_time,fr,total_time
    # LGN_spike_time:[521,8000], fr:[3840,1]

    

# 设计实验:不同分辨率的图片对计算时间的影响  Experiment_resolution(mode,repeat=3):
#除了分辨率，其他参数也可以用这个函数类似进行实验，但是刺激时间不能用这个函数，因为刺激时间不同LGN_spike_time的维数不同，不能直接reshape
def Experiment_resolution(mode,repeat=10):
    '''
    以下参数虽然都是固定的(模拟时间1s,frameRate=1)需要的话可以修改
    mode:grey:灰度图;grating:静态图片;driftgrating:动态图片
    保存数据的类型都是numpy数组,形状如下
    resolution:(len(resolution),1)
    lgn_spike:(len(resolution),repeat,512,8000)
    v1_fr:(len(resolution),repeat,3840)
    run_time:(len(resolution),repeat)记录运行时间
    '''
    P=np.array([0.5*np.pi])
    SF = np.arange(36,40,10)#36
    D = np.array([np.pi/6])  # 方向范围[0,pi]
    resolution=[16,32,64,80,128,200,256]
    lgn_spike=[]
    v1_fr=[]
    run_time=[]
    if mode=="grey":
        C=np.array([0])
        frameRate=1
    elif mode == "grating":
        C=np.array([0.45])
        frameRate=1
    else:
        C=np.array([0.45])
        frameRate=96
        
    for index, size in enumerate(resolution):
        for i in range(repeat):
            LGN_spike_time,fr,total_time=Run_Our_Model(frameRate, 1,index+1,C,P,SF,D,size,f'static_color-grid_size{size}')
            lgn_spike.append(LGN_spike_time)
            v1_fr.append(fr)
            run_time.append(total_time)
    
    resolution, lgn_spike, v1_fr, run_time = map(np.array, [resolution, lgn_spike, v1_fr, run_time])
    lgn_spike=lgn_spike.reshape(resolution.shape[0],repeat, 512, -1)
    v1_fr=v1_fr.reshape(resolution.shape[0],repeat, -1)
    run_time=run_time.reshape(resolution.shape[0],repeat, -1)

    # if mode=="grey":
    np.savez(img_save_path+f'resolution_{mode}.npz',resolution=resolution, lgn_spike=lgn_spike, v1_fr=v1_fr, run_time=run_time)
    # elif mode == "grating":
    #     np.savez(img_save_path+'static_color-grid_resolution.npz',resolution=resolution, lgn_spike=lgn_spike,v1_fr=v1_fr, run_time=run_time)
    # else:
    #     np.savez(img_save_path+'static_color-driftgrid_resolution.npz',resolution=resolution, lgn_spike=lgn_spike,v1_fr=v1_fr, run_time=run_time)
    # return resolution, lgn_spike, v1_fr, run_time

# 设计实验:不同刺激时长对计算时间的影响  
def Experiment_Simulation_duration(mode,repeat=10):
    '''
    以下参数虽然都是固定的(模拟时间1s,frameRate=1)需要的话可以修改
    保存数据的类型都是numpy数组,形状如下
    Simulation_durations:(len(Simulation_durations),1)
    lgn_spike_array:大小为(len(Simulation_durations)*repeat,)的数组,每一项是一个(512,nt)的数组
    v1_fr:(len(Simulation_durations),repeat,3840)
    run_time:(len(Simulation_durations),repeat)记录运行时间
    '''
    P=np.array([0.5*np.pi])
    SF = np.arange(36,40,10)#36
    D = np.array([np.pi/6])  # 方向范围[0,pi]
    resolution=128
    Simulation_duration_list=[1,2,3,4,5,10,15,20]
    lgn_spike=[]
    v1_fr=[]
    run_time=[]
    # C=np.array([0.45])
    if mode=="grey":
        C=np.array([0])
        frameRate=1
    elif mode == "grating":
        C=np.array([0.45])
        frameRate=1
    else:
        C=np.array([0.45])
        frameRate=96
        
    for index, Simulation_duration in enumerate(Simulation_duration_list):
        for i in range(repeat):
            LGN_spike_time,fr,total_time=Run_Our_Model(frameRate,Simulation_duration,index+1,C,P,SF,D,resolution,f'static_color-grid_time{Simulation_duration}')
            lgn_spike.append(LGN_spike_time)
            v1_fr.append(fr)
            run_time.append(total_time)
    Simulation_duration_array, v1_fr, run_time = map(np.array, [Simulation_duration_list, v1_fr, run_time])   
    v1_fr=v1_fr.reshape(Simulation_duration_array.shape[0],repeat, -1)
    run_time=run_time.reshape(Simulation_duration_array.shape[0],repeat, -1)

    lgn_spike_array = np.empty(Simulation_duration_array.shape[0]*repeat, dtype=object)

    # 逐个将原始列表中的二维数组添加到空数组中
    for i in range(Simulation_duration_array.shape[0]*repeat):
        lgn_spike_array[i] = lgn_spike[i]

    np.savez(img_save_path+f'Simulation_duration_{mode}.npz',Simulation_durations=Simulation_duration_array, lgn_spike_array=lgn_spike_array, v1_fr=v1_fr, run_time=run_time)
    
    # return Simulation_durations, lgn_spike_array, v1_fr, run_time

#计算lgn_fr,可以去掉指定时间之前的数据，如果自变量（parameter）是刺激时长的话需要单独处理
#读取文件与上面实验是生成的文件名一致
def compute_lgn_fr(cut_off_time,mode,parameter,STD):
    '''
    parameter=="Simulation_durations":说明Simulation_durations是变量,需要单独处理lgn_spike
    cut_off_time:截断时间,即不考虑该时间之前的spike,单位ms
    lgn_fr:(len(resolution), repeat, 512)  eg:(7,10,512)
    v1_fr:(len(resolution), repeat, 3840)
    '''
    dt=0.125
    if parameter=="Simulation_duration":
        with np.load(img_save_path+f'Simulation_duration_{mode}.npz',allow_pickle=True) as data:
            Simulation_durations = data['Simulation_durations']
            lgn_spike_array = data['lgn_spike_array']
            v1_fr = data['v1_fr']
            run_time = data['run_time']
        lgn_fr_list=[]
        print(lgn_spike_array.shape[0])
        for i in range(lgn_spike_array.shape[0]):
            lgn_spike_single=lgn_spike_array[i]
            time_ms=lgn_spike_single.shape[1]*dt  #模拟时长(ms)
            cut=cut_off_time/dt #(cut_off_time/(nt*dt))*nt  
            #截断的模拟次数，例如模拟时长1s时模拟8000次，cut_off_time设置为200ms,那么只计算后6400次的spike
            lgn_fr_single=np.sum(lgn_spike_single[:,int(cut):],axis=1)/((time_ms-cut_off_time)/1000) 
            lgn_fr_list.append(lgn_fr_single)
        lgn_fr=np.array(lgn_fr_list).reshape(Simulation_durations.shape[0],int(len(lgn_fr_list)/Simulation_durations.shape[0]), -1)  
    else:
        with np.load(img_save_path+f'resolution_{mode}.npz') as data:
            resolution = data['resolution']
            lgn_spike = data['lgn_spike']
            v1_fr = data['v1_fr']
            run_time = data['run_time']

        time_ms=lgn_spike.shape[3]*dt  #模拟时长(ms)
        cut=cut_off_time/dt #(cut_off_time/(nt*dt))*nt  
        #截断的模拟次数，例如模拟时长1s时模拟8000次，cut_off_time设置为200ms,那么只计算后6400次的spike
        lgn_fr=np.sum(lgn_spike[:,:,:,int(cut):],axis=3)/((time_ms-cut_off_time)/1000) #(2, 3, 512, nt)->(2,3,512)
        print(f"lgn_fr={lgn_fr.shape}")

    lgn_fr_max=np.max(np.mean(lgn_fr,axis=1),axis=1)#(7, 10, 512)->(7,512)->(7,)
    v1_fr_max=np.max(np.mean(v1_fr,axis=1),axis=1)
    lgn_fr_mean=np.mean(np.mean(lgn_fr,axis=1),axis=1)
    v1_fr_mean=np.mean(np.mean(v1_fr,axis=1),axis=1)

    run_time_mean=np.mean(np.mean(run_time,axis=1),axis=1)

    lgn_std=np.std(np.mean(lgn_fr,axis=2),axis=1)#(7, 10, 512)->(7,10)->(7,)
    v1_std=np.std(np.mean(v1_fr,axis=2),axis=1)
    lgn_std_single=np.std(lgn_fr,axis=1)#(7, 10, 512)->(7,512)
    v1_std_single=np.std(v1_fr,axis=1)  

    if parameter=="Simulation_duration":
        if STD==True:
            return Simulation_durations, lgn_std, v1_std ,lgn_std_single, v1_std_single
        else:    
            return Simulation_durations, lgn_fr_max, v1_fr_max, lgn_fr_mean, v1_fr_mean, run_time_mean
    else:
        if STD==True:
            return resolution, lgn_std, v1_std ,lgn_std_single, v1_std_single
        else:    
            return resolution, lgn_fr_max, v1_fr_max, lgn_fr_mean, v1_fr_mean, run_time_mean

def plot_time_resolution(resolution,lgn_fr_max,v1_fr_max,lgn_fr_mean,v1_fr_mean,run_time_mean,pic_name):
    # 为了方便，将所有向量转换为一维数组
    resolution = resolution.flatten()
    lgn_fr_vector = lgn_fr_mean.flatten()
    v1_fr_vector = v1_fr_mean.flatten()
    lgn_max_vector = lgn_fr_max.flatten()
    v1_max_vector = v1_fr_max.flatten()
    run_time_vector = run_time_mean.flatten()

    #颜色
    # color=['#8ECFC9','#FFBE7A','#FA7F6F','#82B0D2','#BEB8DC','#E7DAD2']
    color=['#2878b5','#c82423','#ff8884','#9ac9db','#f8ac8c',]

    # 创建一个新的图形
    fig, ax1 = plt.subplots(figsize=(8, 6))
    plt.xticks(resolution)

    # 创建第一个纵坐标
    ax1.set_xlabel('resolution')
    ax1.set_ylabel('lgn_fr(Hz)', fontsize=14, labelpad=10,color=color[0]).set_rotation(0)
    ax1.plot(resolution, lgn_fr_vector, color=color[0], marker='o')
    ax1.tick_params(axis='y', labelcolor=color[0])
    ax1.set_ylim([0, np.max(lgn_fr_vector)*1.2])

    # 创建第二个纵坐标
    ax2 = ax1.twinx()
    ax2.set_ylabel('v1_fr(Hz)', fontsize=14, labelpad=30,color=color[1]).set_rotation(0)
    ax2.plot(resolution, v1_fr_vector, color=color[1], marker='o')
    ax2.tick_params(axis='y', labelcolor=color[1])
    ax2.set_ylim([0, np.max(v1_fr_vector)*1.2])

    # 创建第三个纵坐标
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(('axes', 1.2))
    ax3.set_ylabel('run_time(s)', fontsize=14, labelpad=40,color=color[2]).set_rotation(0)
    ax3.plot(resolution, run_time_vector, color=color[2], marker='o')
    ax3.tick_params(axis='y', labelcolor=color[2])
    ax3.set_ylim([0, np.max(run_time_vector)*1.1])

    # 创建第四个纵坐标  lgn_max全为0，不再展示
    ax4 = ax1.twinx()
    ax4.spines.right.set_position(('axes', 1.3))
    ax4.set_ylabel('lgn_max(Hz)', fontsize=14, labelpad=40,color=color[4]).set_rotation(0)
    ax4.plot(resolution, lgn_max_vector, color=color[4], marker='o')
    ax4.tick_params(axis='y', labelcolor=color[4])
    ax4.set_ylim([0, np.max(lgn_max_vector)+0.1])
    # ax4.tick_params(labelright=False)  #隐藏纵坐标轴

    # 创建第五个纵坐标
    ax5 = ax1.twinx()
    ax5.spines.right.set_position(('axes', 1.4))
    ax5.set_ylabel('v1_std(Hz)', fontsize=14, labelpad=40,color=color[3]).set_rotation(0)
    ax5.plot(resolution, v1_max_vector, color=color[3], marker='o')
    ax5.tick_params(axis='y', labelcolor=color[3])
    ax5.set_ylim([0, np.max(v1_max_vector)*1.3])
    # ax5.tick_params(labelright=False)

    # 显示每个点的值
    for i in range(len(resolution)):

        ax1.annotate(f"{lgn_fr_vector[i]:.2f}", (resolution[i], lgn_fr_vector[i]), xytext=(resolution[i]-6, lgn_fr_vector[i]*1.03),color=color[0])

        ax3.annotate(f"{run_time_vector[i]:.2f}", (resolution[i], run_time_vector[i]), xytext=(resolution[i], run_time_vector[i]*1.02),color=color[2])

        ax4.annotate(f"{lgn_max_vector[i]:.2f}", (resolution[i], lgn_max_vector[i]), xytext=(resolution[i], lgn_max_vector[i]),color=color[4])

    if i==0:
        ax2.annotate(f"{v1_fr_vector[i]:.2f}", (resolution[i], v1_fr_vector[i]), xytext=(resolution[i]+3, v1_fr_vector[i]*0.95),color=color[1])
        ax5.annotate(f"{v1_max_vector[i]:.4f}", (resolution[i], v1_max_vector[i]), xytext=(resolution[i]-8, v1_max_vector[i]*0.8),color=color[3])
    else:
        ax2.annotate(f"{v1_fr_vector[i]:.2f}", (resolution[i], v1_fr_vector[i]), xytext=(resolution[i], v1_fr_vector[i]*0.95),color=color[1])
        ax5.annotate(f"{v1_max_vector[i]:.4f}", (resolution[i], v1_max_vector[i]), xytext=(resolution[i]-8, v1_max_vector[i]*0.92),color=color[3])
    

    # 显示图例
    legend_labels = ['lgn_fr(Hz)', 'v1_fr(Hz)', 'run_time(s)','lgn_max(Hz)', 'v1_max(Hz)']
    plt.legend([ax1.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0],ax4.get_lines()[0],ax5.get_lines()[0]], legend_labels, loc='center', bbox_to_anchor=(0.5, 0.1))

    plt.savefig(os.path.join(img_save_path, f'{pic_name}'))#第一个是指存储路径，第二个是图片名字
    plt.close()

def plot_heatmap_resolution(resolution,lgn_fr_max,v1_fr_max,lgn_fr_mean,v1_fr_mean,run_time_mean,lgn_std, v1_std,pic_name):
   
    # plt.errorbar(resolution, lgn_fr_max, yerr=lgn_std, label='lgn_fr_max', fmt='-o')
    # plt.errorbar(resolution, v1_fr_max, yerr=v1_std, label='v1_fr_max', fmt='-o')
    # plt.errorbar(resolution, lgn_fr_mean, yerr=lgn_std, label='lgn_fr_mean', fmt='-o')
    # plt.errorbar(resolution, v1_fr_mean, yerr=v1_std, label='v1_fr_mean', fmt='-o')
    # plt.errorbar(resolution, run_time_mean, label='run_time_mean', fmt='-o')

    # plt.xlabel('Resolution')
    # plt.ylabel('Value')
    # plt.title('Relationship between Variables and Resolution')
    # plt.legend()

    x = resolution
    plt.scatter(x, np.log(lgn_fr_max), label='lgn_fr_max')
    plt.scatter(x, np.log(v1_fr_max), label='v1_fr_max')
    plt.scatter(x, np.log(lgn_fr_mean), label='lgn_fr_mean')
    plt.scatter(x, np.log(v1_fr_mean), label='v1_fr_mean')
    plt.scatter(x, np.log(run_time_mean), label='run_time_mean')
    plt.scatter(x, np.log(lgn_std), label='lgn_std')
    plt.scatter(x, np.log(v1_std), label='v1_std')

    plt.xlabel('Resolution')
    plt.ylabel('Value (log scale)')
    plt.title('Relationship between Variables and Resolution')
    plt.legend()
    plt.show()

    plt.savefig(os.path.join(img_save_path, f'{pic_name}_heatmap'))#第一个是指存储路径，第二个是图片名字
    plt.close()


def plot_time_Simulation_durations(Simulation_duration,lgn_fr_max,v1_fr_max,lgn_fr_mean,v1_fr_mean,run_time_mean,pic_name):
    # 为了方便，我们将所有向量转换为一维数组
    Simulation_duration = Simulation_duration.flatten()
    lgn_fr_vector = lgn_fr_mean.flatten()
    v1_fr_vector = v1_fr_mean.flatten()
    lgn_max_vector = lgn_fr_max.flatten()
    v1_max_vector = v1_fr_max.flatten()
    run_time_vector = run_time_mean.flatten()

    #颜色
    # color=['#8ECFC9','#FFBE7A','#FA7F6F','#82B0D2','#BEB8DC','#E7DAD2']
    color=['#2878b5','#c82423','#ff8884','#9ac9db','#f8ac8c',]

    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.xticks(Simulation_duration)

    # 创建第一个纵坐标
    ax.set_xlabel('Simulation_duration_time(s)')
    ax.set_ylabel('lgn_fr(Hz)', fontsize=14, labelpad=10,color=color[0]).set_rotation(0)
    ax.plot(Simulation_duration, lgn_fr_vector, color=color[0], marker='o')
    ax.tick_params(axis='y', labelcolor=color[0])
    ax.set_ylim([0, np.max(lgn_fr_vector)*1.2])

    # 创建第二个纵坐标
    ax2 = ax.twinx()
    ax2.set_ylabel('v1_fr(Hz)', fontsize=14, labelpad=30,color=color[1]).set_rotation(0)
    ax2.plot(Simulation_duration, v1_fr_vector, color=color[1], marker='o')
    ax2.tick_params(axis='y', labelcolor=color[1])
    ax2.set_ylim([0, np.max(v1_fr_vector)*1.2])

    # 创建第三个纵坐标
    ax3 = ax.twinx()
    ax3.spines.right.set_position(('axes', 1.2))
    ax3.set_ylabel('run_time(s)', fontsize=14, labelpad=40,color=color[2]).set_rotation(0)
    ax3.plot(Simulation_duration, run_time_vector, color=color[2], marker='o')
    ax3.tick_params(axis='y', labelcolor=color[2])
    ax3.set_ylim([0, np.max(run_time_vector)*1.2])

    # 创建第四个纵坐标  lgn_max全为0，不再展示
    ax4 = ax.twinx()
    ax4.spines.right.set_position(('axes', 1.3))
    ax4.set_ylabel('lgn_max(Hz)', fontsize=14, labelpad=40,color=color[4]).set_rotation(0)
    ax4.plot(Simulation_duration, lgn_max_vector, color=color[4], marker='o')
    ax4.tick_params(axis='y', labelcolor='yellow')
    ax4.set_ylim([0, np.max(lgn_max_vector)+0.1])
    # ax4.tick_params(labelright=False)  #隐藏纵坐标轴

    # 创建第五个纵坐标
    ax5 = ax.twinx()
    ax5.spines.right.set_position(('axes', 1.4))
    ax5.set_ylabel('v1_max(Hz)', fontsize=14, labelpad=40,color=color[3]).set_rotation(0)
    ax5.plot(Simulation_duration, v1_max_vector, color=color[3], marker='o')
    ax5.tick_params(axis='y', labelcolor=color[3])
    ax5.set_ylim([0, np.max(v1_max_vector)*1.3])
    # ax5.tick_params(labelright=False)

    # 显示每个点的值，为了展示出来好看需要挨个儿调文字位置、箭头形状位置颜色等。
    for i in range(len(Simulation_duration)):

        if i % 2 == 0:
            ax.annotate(f"{lgn_fr_vector[i]:.2f}", (Simulation_duration[i], lgn_fr_vector[i]), xytext=(Simulation_duration[i], lgn_fr_vector[i]*1.12),arrowprops=dict(facecolor=color[0], arrowstyle='fancy'),color=color[0])
            ax2.annotate(f"{v1_fr_vector[i]:.2f}", (Simulation_duration[i], v1_fr_vector[i]), xytext=(Simulation_duration[i], v1_fr_vector[i]*0.9),arrowprops=dict(facecolor=color[1], arrowstyle='simple'),color=color[1])
        else:
            ax.annotate(f"{lgn_fr_vector[i]:.2f}", (Simulation_duration[i], lgn_fr_vector[i]), xytext=(Simulation_duration[i]-0.4, lgn_fr_vector[i]*1.05),arrowprops=dict(facecolor=color[0], arrowstyle='fancy'),color=color[0])
            ax2.annotate(f"{v1_fr_vector[i]:.2f}", (Simulation_duration[i], v1_fr_vector[i]), xytext=(Simulation_duration[i], v1_fr_vector[i]*0.85),arrowprops=dict(facecolor=color[1], arrowstyle='simple'),color=color[1])

        ax3.annotate(f"{run_time_vector[i]:.2f}", (Simulation_duration[i], run_time_vector[i]), xytext=(Simulation_duration[i], run_time_vector[i]-3),color=color[2])

        ax4.annotate(f"{lgn_max_vector[i]:.2f}", (Simulation_duration[i], lgn_max_vector[i]), xytext=(Simulation_duration[i], lgn_max_vector[i]),color=color[4])

        if i==0:
            ax5.annotate(f"{v1_max_vector[i]:.4f}", (Simulation_duration[i], v1_max_vector[i]), xytext=(Simulation_duration[i]*0.8, v1_max_vector[i]*0.85),arrowprops=dict(facecolor=color[3], arrowstyle='wedge'),color=color[3])
        else:
            ax5.annotate(f"{v1_max_vector[i]:.4f}", (Simulation_duration[i], v1_max_vector[i]), xytext=(Simulation_duration[i]*0.95, v1_max_vector[i]*1.05),color=color[3])

    # 显示图例
    legend_labels = ['lgn_fr(Hz)', 'v1_fr(Hz)', 'run_time(s)','lgn_max(Hz)','v1_max(Hz)']
    plt.legend([ax.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0], ax4.get_lines()[0], ax5.get_lines()[0]], legend_labels, loc='center', bbox_to_anchor=(0.5, 0.1))
    plt.title(f'{pic_name}')
    plt.savefig(os.path.join(img_save_path, f'{pic_name}'))#第一个是指存储路径，第二个是图片名字
    plt.close()


if __name__ == "__main__" :
    modes=['grey','grating','driftgrating']
    cut_off_time=0
    parameter='resolution'
    # parameter='Simulation_duration'
    experiment = False #生成数据，运行一次即可，后面计算可以改为False
    
    for mode in modes:
        pic_name=f'{parameter}_{mode}'
        if parameter=='resolution':
            if experiment==True:
                Experiment_resolution(mode,repeat=10)
            resolution,lgn_fr_max, v1_fr_max, lgn_fr_mean, v1_fr_mean,run_time_mean=compute_lgn_fr(cut_off_time,mode,parameter,False)
            _, lgn_std, v1_std ,lgn_std_single, v1_std_single=compute_lgn_fr(cut_off_time,mode,parameter,True)
            # plot_time_resolution(resolution,lgn_fr_max,v1_fr_max,lgn_fr_mean,v1_fr_mean,run_time_mean,pic_name)
            plot_heatmap_resolution(resolution,lgn_fr_max,v1_fr_max,lgn_fr_mean,v1_fr_mean,run_time_mean,lgn_std, v1_std,pic_name)
        else:
            if experiment==True:
                Experiment_Simulation_duration(mode,repeat=10)
            Simulation_duration,lgn_fr_max, v1_fr_max, lgn_fr_mean, v1_fr_mean,run_time_mean=compute_lgn_fr(cut_off_time,mode,parameter,False)
            plot_time_Simulation_durations(Simulation_duration,lgn_fr_max,v1_fr_max,lgn_fr_mean,v1_fr_mean,run_time_mean,pic_name)
    print('--------------finished-------------------')