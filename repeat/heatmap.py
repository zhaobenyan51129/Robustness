
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from repeat_dw import read_output
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示
import warnings
warnings.filterwarnings('ignore')

#读取所有time所有contrast重复10次的数据，输出：fr_vector(5,10,10,3840),error_vector(5,10,10,3840)
def merge_data_fr():
    fr_list=[]
    for time in times:
        for k in range(n_pic):
            for i in range(repeat):
                fr_vector=read_output(dir+'/contrast{}/fr_time{}.npz'.format(k+1,time))
                fr_list.append(fr_vector[i])
    fr_vector = np.array(fr_list).reshape(len(times),n_pic, repeat, -1)
    error_list=[]
    for time in times:
        fr_tensor_time = fr_vector[time-1] #(10,10,3840)
        error_pic=[]
        for i in range(10):#n_pic):
            fr= fr_tensor_time[i]#(10,3840)
            error=np.abs(fr-fr[0])#(10,3840)
            error_pic.append(error)
        error_list.append(error_pic)
    error_vector=np.array(error_list)
    return fr_vector, error_vector

##读取固定time所有contrast第一次重复的数据，输出：长为10（n_pic)的列表，每一项为（521，nt)的数组
#时间不同nt不同，因此不能所有时间同时读取
def merge_data_lgn(time,repeat):
    lgn_list=[] 
    for k in range(n_pic):
        lgn_vector=read_output(dir+'/contrast{}/lgn_time{}.npz'.format(k+1,time)) #521*nt
        # lgn_list.append(lgn_vector[0].flatten())
        lgn_list.append(lgn_vector[repeat])
    # lgn_vector = np.array(lgn_list).reshape(n_pic, -1)
    return lgn_list

#计算不同时间不同contrast的图片,lgn输出的fr（mean),输出为（5，10）的array,保存在二进制文件lgn_fr_vector.npz
def compute_lgn_fr(): 
    lgn_fr_list=[]
    for time in times:
        lgn_vector=np.array(merge_data_lgn(time,0))#(10,512,8000)
        lgn_fr_per_neuro=np.sum(lgn_vector,axis=2)/time #(10,512)
        lgn_fr_mean=np.mean(lgn_fr_per_neuro,axis=1) #(10,) 每张图lgn所有neuro的平均放电率
        lgn_fr_list.append(lgn_fr_mean)
    np.savez(dir+'/lgn_fr_vector',lgn_fr_vector=np.array(lgn_fr_list))

#计算lgn_fr的波动
def compute_lgn_error_fr():
    lgn_list=[]
    for time in times:
        for k in range(n_pic):
            for i in range(repeat):
                lgn_vector=read_output(dir+'/contrast{}/lgn_time{}.npz'.format(k+1,time)) 
                lgn_fr=np.sum(lgn_vector,axis=2)/time #(10,512)
                lgn_list.append(lgn_fr[i])
    lgn_fr_vector = np.array(lgn_list).reshape(len(times),n_pic, repeat, -1)
    error_list=[]
    for time in times:
        fr_tensor_time = lgn_fr_vector[time-1] #(10,10,3840)
        error_pic=[]
        for i in range(10):#n_pic):
            fr= fr_tensor_time[i]#(10,3840)
            error=np.abs(fr-fr[0])#(10,3840)
            error_pic.append(error)
        error_list.append(error_pic)
    lgn_error_vector=np.array(error_list)
    print(f"lgn_fr_vector={lgn_fr_vector.shape}")
    print(f"lgn_error_vector={lgn_error_vector.shape}")
    # return lgn_fr_vector, lgn_error_vector
    np.savez(dir+'/lgn_fr_vector_all',lgn_fr_vector=lgn_fr_vector,lgn_error_vector=lgn_error_vector)

#对不同时间时，重复十次fr及其误差的热图fr_vector为上一个函数读出来的数据（5，10，10，3840）
#画两张4x5的图
def plot_fr_error(fr_vector,STD):
    for time in times:
        fr_tensor = fr_vector[time-1] #(10,10,3840)
        fr_mean = np.mean(fr_tensor,axis=1)
        if STD==False:
            #图1：重复十次，fr的值（按fr_mean上升排序）
            fig1 = plt.figure(figsize=(60,18),dpi=200)
            for i in range(10):#n_pic):
                fig1.add_subplot(2,5,i+1)
                fr_mean_df=pd.DataFrame(fr_mean[i]).T.round(2)
                fr_df = pd.DataFrame(fr_tensor[i])#(10,3840)
                df1=fr_mean_df.append(fr_df, ignore_index=True)
                df1=df1.sort_values(by=0,axis=1,ascending=True)
                df1.columns = df1.loc[0]
                df1 = df1[1:]
                ax1=sns.heatmap(data=df1,
                            cmap=plt.get_cmap('Greens'),#matplotlib中的颜色盘'Greens'
                            vmin=0,#图例（右侧颜色条color bar）中最小显示值 
                            vmax=40,#图例（右侧颜色条color bar）中最大显示值
                        )
                plt.title("fr:contrast={}".format((i+1)/20),fontsize = 20)
                ax1.invert_yaxis()  #将y轴顺序颠倒
                ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)  #旋转y轴坐标数据的方向
                # plt.title("fr:contrast={}".format((i+1)/20),fontsize = 20)
            plt.suptitle('fr,time={}s'.format(time), ha = 'left',fontsize = 30, weight = 'extra bold')
            plt.savefig(os.path.join(dir+'/heatmap', 'fr,time={}s'.format(time)))#第一个是指存储路径，第二个是图片名字
            plt.close()

            #图2，重复十次error的值（按fr_mean上升排序）
            fig2 = plt.figure(figsize=(60,18),dpi=200)
            for i in range(10):#n_pic):
                fig2.add_subplot(2,5,i+1)
                fr_mean_df=pd.DataFrame(fr_mean[i]).T.round(2)
                fr_df = pd.DataFrame(fr_tensor[i])#(10,3840)
                error_df=(fr_df-fr_df.iloc[0]).abs()
                df2=fr_mean_df.append(error_df, ignore_index=True)
                df2=df2.sort_values(by=0,axis=1,ascending=True)
                df2.columns = df2.loc[0]
                df2 = df2[1:]
                ax2=sns.heatmap(data=df2,
                            cmap=plt.get_cmap('Greens'),
                            vmin=0,
                            vmax=2,
                        )
                ax2.invert_yaxis()  #将y轴顺序颠倒
                ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)  #旋转y轴坐标数据的方向
                plt.title("error:contrast={}".format((i+1)/20),fontsize = 20)
            plt.suptitle('error,sorted by fr,time={}s'.format(time), ha = 'left',fontsize = 30, weight = 'extra bold')
            plt.savefig(os.path.join(dir+'/heatmap', 'error,sorted by fr,time={}s'.format(time)))#第一个是指存储路径，第二个是图片名字
            plt.close()
            
            #图3：error,以error_mean排序
            fig3 = plt.figure(figsize=(60,18),dpi=200)
            for i in range(10):#n_pic):
                fig3.add_subplot(2,5,i+1)
                fr_mean_df=pd.DataFrame(fr_mean[i]).T.round(2)
                fr_df = pd.DataFrame(fr_tensor[i])#(10,3840)
                error_df=(fr_df-fr_df.iloc[0]).abs().round(2)
                error_df.loc[10] = error_df.apply(lambda x: x[1:10].mean(),axis=0).round(2)
                df2=error_df.sort_values(by=10,axis=1,ascending=True)
                df2.columns = df2.loc[10]
                df2 = df2[:10]
                df2.index = np.arange(1,11)
                ax2=sns.heatmap(data=df2,
                            cmap=plt.get_cmap('Greens'),
                            vmin=0,
                            vmax=2,
                        )
                ax2.invert_yaxis()  #将y轴顺序颠倒
                ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)  #旋转y轴坐标数据的方向
                plt.title("error:contrast={}".format((i+1)/20),fontsize = 20)
            plt.suptitle('error,sorted by mean_error,time={}s'.format(time), ha = 'left',fontsize = 30, weight = 'extra bold')
            plt.savefig(os.path.join(dir+'/heatmap', 'error,sorted by mean_error,time={}s'.format(time)))#第一个是指存储路径，第二个是图片名字
            plt.close()
        else:
            #图4：重复十次std的值（按fr_mean上升排序)
            fig4 = plt.figure(figsize=(60,18),dpi=200)
            for i in range(10):#n_pic):
                fig4.add_subplot(2,5,i+1)
                fr_mean_df=pd.DataFrame(fr_mean[i]).T.round(2)
                fr_df = pd.DataFrame(fr_tensor[i])#(10,3840)
                std_df=fr_df.std()
                print(f"std_df.shape={std_df.shape}")
                df4=fr_mean_df.append(std_df, ignore_index=True)
                df4=df4.sort_values(by=0,axis=1,ascending=True)
                df4.columns = df4.loc[0]
                std_df = df4[1:]
                print(f"std_df.shape={std_df.shape}")
                # 将std_df转换为一个62x62的矩阵
                # 将std_df转换为一个长度为3840的数组
                std_arr = std_df.values.flatten()

                # 如果std_arr的长度不足3844个，使用pad方法在右下角填充0
                if len(std_arr) < 3844:
                    n_missing = 3844 - len(std_arr)
                    std_arr = np.pad(std_arr, pad_width=(0, n_missing), mode='constant', constant_values=0)

                # 将std_arr重塑为一个62x62的矩阵
                matrix = std_arr.reshape((62, 62))


                ax4=sns.heatmap(data=matrix,
                            cmap=plt.get_cmap('Greens'),
                            vmin=0,
                            vmax=1,
                        )
                # ax4.invert_yaxis()  #将y轴顺序颠倒
                ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)  #旋转y轴坐标数据的方向
                plt.title(f"contrast={(i+1)/20}_time={time}s_std",fontsize = 20)
            plt.suptitle(f"heatmap_std_{time}", ha = 'left',fontsize = 30, weight = 'extra bold')
            plt.savefig(os.path.join(dir+'/heatmap', f"heatmap_std_{time}"))#第一个是指存储路径，第二个是图片名字
            plt.close()

            #图5：重复十次mean的值（按fr_mean上升排序)
            fig5 = plt.figure(figsize=(60,18),dpi=200)
            for i in range(10):#n_pic):
                fig5.add_subplot(2,5,i+1)
                fr_mean_df=pd.DataFrame(fr_mean[i]).T.round(2)
                fr_df = pd.DataFrame(fr_tensor[i])#(10,3840)
                mean_df=fr_df.mean()
                print(f"mean_df.shape={mean_df.shape}")
                df5=fr_mean_df.append(mean_df, ignore_index=True)
                df5=df5.sort_values(by=0,axis=1,ascending=True)
                df5.columns = df5.loc[0]
                mean_df = df5[1:]
                print(f"mean_df.shape={mean_df.shape}")
    
                mean_arr = mean_df.values.flatten()

                # 如果std_arr的长度不足3844个，使用pad方法在右下角填充0
                if len(mean_arr) < 3844:
                    n_missing = 3844 - len(mean_arr)
                    mean_arr = np.pad(mean_arr, pad_width=(0, n_missing), mode='constant', constant_values=0)

                # 将std_arr重塑为一个62x62的矩阵
                matrix1 = mean_arr.reshape((62, 62))


                ax5=sns.heatmap(data=matrix1,
                            cmap=plt.get_cmap('Greens'),
                            vmin=0,
                            vmax=40,
                        )
                # ax5.invert_yaxis()  #将y轴顺序颠倒
                ax5.set_yticklabels(ax5.get_yticklabels(), rotation=0)  #旋转y轴坐标数据的方向
                plt.title(f"contrast={(i+1)/20}_time={time}s_mean",fontsize = 20)
            plt.suptitle(f"heatmap_mean_{time}", ha = 'left',fontsize = 30, weight = 'extra bold')
            plt.savefig(os.path.join(dir+'/heatmap', f"heatmap_mean_{time}"))#第一个是指存储路径，第二个是图片名字
            plt.close()

            #图6：重复十次max的值（按fr_mean上升排序)
            fig6 = plt.figure(figsize=(60,18),dpi=200)
            for i in range(10):#n_pic):
                fig6.add_subplot(2,5,i+1)
                fr_mean_df=pd.DataFrame(fr_mean[i]).T.round(2)
                fr_df = pd.DataFrame(fr_tensor[i])#(10,3840)
                max_df=fr_df.max()
                print(f"max_df.shape={max_df.shape}")
                df6=fr_mean_df.append(max_df, ignore_index=True)
                df6=df6.sort_values(by=0,axis=1,ascending=True)
                df6.columns = df6.loc[0]
                max_df = df6[1:]
                print(f"max_df.shape={max_df.shape}")
                # 将std_df转换为一个62x62的矩阵
                # 将std_df转换为一个长度为3840的数组
                max_arr = max_df.values.flatten()

                # 如果std_arr的长度不足3844个，使用pad方法在右下角填充0
                if len(max_arr) < 3844:
                    n_missing = 3844 - len(max_arr)
                    max_arr = np.pad(max_arr, pad_width=(0, n_missing), mode='constant', constant_values=0)

                # 将std_arr重塑为一个62x62的矩阵
                matrix2 = max_arr.reshape((62, 62))


                ax6=sns.heatmap(data=matrix2,
                            cmap=plt.get_cmap('Greens'),
                            vmin=0,
                            vmax=40,
                        )
                # ax6.invert_yaxis()  #将y轴顺序颠倒
                ax6.set_yticklabels(ax6.get_yticklabels(), rotation=0)  #旋转y轴坐标数据的方向
                plt.title(f"contrast={(i+1)/20}_time={time}s_max",fontsize = 20)
            plt.suptitle(f"heatmap_max_{time}", ha = 'left',fontsize = 30, weight = 'extra bold')
            plt.savefig(os.path.join(dir+'/heatmap', f"heatmap_max_{time}"))#第一个是指存储路径，第二个是图片名字
            plt.close()
            

#fr_mean和fr_error随time和contrast的变化，fr_vector:(5,10,10,3840) mode:对3840个neuron取二范数还是均值还是最大值
def plot_time_contrast(fr_vector,error_vector,mode):
    #先对3480个nuero取平均再对重复次数取平均
    fr_mean=np.mean(fr_vector,axis=3)  #(5,10,10)
    fr_mean=np.mean(fr_mean,axis=2)    #(5,10)
    error_mean=np.mean(error_vector,axis=3)  #(5,10,10)
    error_mean=np.mean(error_mean,axis=2)    #(5,10)
    #先对3480个nuero取二范再对重复次数取平均
    fr_norm=np.linalg.norm(fr_vector,axis=3)
    fr_norm=np.mean(fr_norm,axis=2)
    error_norm=np.linalg.norm(error_vector,axis=3)  #(5,10,10)
    error_norm=np.mean(error_norm,axis=2)
    #先对3480个nuero取max再对重复次数取平均
    fr_max=np.max(fr_vector,axis=3)
    fr_max=np.mean(fr_max,axis=2)
    error_max=np.max(error_vector,axis=3)  #(5,10,10)
    error_max=np.mean(error_max,axis=2)  #（5，10）

    if mode =='mean':
        df_fr = pd.DataFrame(fr_mean)
        df_error=pd.DataFrame(error_mean)
    elif mode =='norm':
        df_fr = pd.DataFrame(fr_norm)
        df_error=pd.DataFrame(error_norm)
    else:
        df_fr = pd.DataFrame(fr_max)
        df_error=pd.DataFrame(error_max)
    #对fr
    fig = plt.figure(figsize=(10,3),dpi=200)
    fig.add_subplot(1,2,1)
    df_fr.index=[1,2,3,4,5]
    df_fr.columns=["%.2f"%i for i in list(np.arange(0.05,0.55,0.05))]
    ax1=sns.heatmap(data=df_fr,  #数据（dataframe型）（5，10）
                    cmap=plt.get_cmap('Greens'),  #颜色风格
                    annot=True,  #是否显示每个方块的数据
                    fmt=".3f",   #每个方块的数据显示三位小数
                    annot_kws={'size':4,'weight':'normal', 'color':'black'}  #每个方块的数据的大小颜色
                )
    ax1.invert_yaxis()  #将y轴顺序颠倒
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)  #旋转x轴坐标数据的方向
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)  #旋转x轴坐标数据的方向
    ax1.tick_params(labelsize=4) # 设置坐标轴数字的字号
    ax1.set_xlabel('contrast',fontsize=5) 
    ax1.set_ylabel('time/s',fontsize=5 )# 设置y轴标签的字体大小和字体颜色
    cbar1 = ax1.collections[0].colorbar #获取图例
    cbar1.ax.tick_params(labelsize=3) #设置图例的字号
    plt.title("fr",fontsize = 6)
    #对error
    fig.add_subplot(1,2,2)
    df_error.index=[1,2,3,4,5]
    df_error.columns=["%.2f"%i for i in list(np.arange(0.05,0.55,0.05))]
    ax2=sns.heatmap(data=df_error,
                    cmap=plt.get_cmap('Greens'),
                    annot=True,
                    fmt=".3f",
                    annot_kws={'size':4,'weight':'normal', 'color':'black'},
                )
    ax2.invert_yaxis()  #将y轴顺序颠倒
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)  #旋转x轴坐标数据的方向
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)  #旋转x轴坐标数据的方向
    ax2.tick_params(labelsize=4) # 设置坐标轴数字的字号
    ax2.set_xlabel('contrast',fontsize=5) 
    ax2.set_ylabel('time/s',fontsize=5 )# 设置y轴标签的字体大小和字体颜色
    cbar2 = ax2.collections[0].colorbar #获取图例
    cbar2.ax.tick_params(labelsize=3) #设置图例的字号
    plt.title("error",fontsize = 6)
    plt.suptitle('time_contrast_{}'.format(mode), ha = 'left',fontsize = 8, weight = 'extra bold')
    plt.savefig(os.path.join(dir+'/heatmap', 'lgn_time_contrast_{}'.format(mode)))#第一个是指存储路径，第二个是图片名字
    plt.close()

#不同时间和contrast下第一次重复,lgn_fr_mean
def plot_lgn_fr():
    with np.load(dir+'/lgn_fr_vector.npz') as f:
        lgn_fr_vector=f['lgn_fr_vector']
    df=pd.DataFrame(lgn_fr_vector)
    df.columns=["%.2f"%i for i in list(np.arange(0.05,0.55,0.05))]
    df.index=[1,2,3,4,5]
    # df=df.apply(lambda x: x*100).round(4)
    fig = plt.figure(figsize=(5,3),dpi=200)
    ax=sns.heatmap(data=df,
                    cmap=plt.get_cmap('Greens'),
                    annot=True,
                    fmt=".2f",
                    # fmt='%.4f%%' %(df*100),
                    annot_kws={'size':4,'weight':'normal', 'color':'black'},
                )
    ax.invert_yaxis()  #将y轴顺序颠倒
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  #旋转x轴坐标数据的方向
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  #旋转y轴坐标数据的方向
    ax.tick_params(labelsize=4) # 设置坐标轴数字的字号
    ax.set_xlabel('contrast',fontsize=5) 
    ax.set_ylabel('time/s',fontsize=5 )# 设置y轴标签的字体大小和字体颜色
    cbar = ax.collections[0].colorbar #获取图例
    cbar.ax.tick_params(labelsize=3) #设置图例的字号
    plt.suptitle('heatmap_lgn', ha = 'left',fontsize = 8, weight = 'extra bold')
    plt.savefig(os.path.join(dir+'/heatmap', 'heatmap_lgn_fr'))#第一个是指存储路径，第二个是图片名字
    plt.close()

def Plot_Log_Log_fig(mode):
    if mode=='lgn':
        with np.load(dir+'/lgn_fr_vector_all.npz') as f:
            fr_vector=f['lgn_fr_vector']#(5, 10, 10, 512)
            # error_vector=f['lgn_error_vector']#(5, 10, 10, 512)
    elif mode=='v1_fr':
        fr_vector,_=merge_data_fr()
    else: #v1_error
        _,fr_vector=merge_data_fr()
    
    fr_mean=np.mean(fr_vector,axis=3)  #(5,10,10)
    fr_mean=np.mean(fr_mean,axis=2)    #(5,10)
    times=np.array([1,2,3,4,5])
    # 对数转换
    time_log = np.log10(times)
    data_log = np.log10(fr_mean)
    # 计算系数
    coefficients = np.polyfit(time_log, data_log, 1)
    print(coefficients)
    fig, ax = plt.subplots(figsize=(5,3.5),dpi=200)
    for i in range(10):
        ax.plot(time_log, data_log[:,i], label='contrast={}'.format((i+1)/20))
        ax.text(time_log[0], data_log[0,i], 'y={}x+{}'.format('{:.4f}'.format(coefficients[0,i]),'{:.4f}'.format(coefficients[1,i])), fontsize=7)  # 在第一个点处添加文本
    ax.set_xlabel('Log(time)',fontsize = 8)
    ax.set_ylabel('Log(fr_mean)',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    # 生成图例并放在右侧
    legend = plt.figlegend(*ax.get_legend_handles_labels(), loc='center right',prop={'size': 8})
    # 调整曲线图和图例图的间距
    fig.subplots_adjust(right=0.75)
    legend.set_bbox_to_anchor((1, 0.5))
    plt.suptitle('log_log_{}'.format(mode), ha = 'left',fontsize = 8, weight = 'extra bold')
    plt.savefig(os.path.join(dir+'/heatmap', 'log_log_{}'.format(mode)))#第一个是指存储路径，第二个是图片名字
    plt.close()


if __name__ == "__main__" :
    times=[1,2,3,4,5]  #运行时间
    n_pic=10      #图片个数
    repeat=10    #重复次数
    dir='/home/zhaobenyan/dataset/output/driftgrating_32x32'    #需要读取的数据所在路径
    fr_vector,error_vector=merge_data_fr()
    plot_fr_error(fr_vector,True)
    # plot_time_contrast(fr_vector,error_vector,mode='norm')
    # plot_time_contrast(fr_vector,error_vector,mode='mean')
    # plot_time_contrast(fr_vector,error_vector,mode='max')
    # compute_lgn_fr() #运行一次即可
    # plot_lgn_fr()
    # compute_lgn_error_fr()
    # with np.load(dir+'/lgn_fr_vector_all.npz') as f:
    #     lgn_fr_vector=f['lgn_fr_vector']
    #     lgn_error_vector=f['lgn_error_vector']
    # plot_time_contrast(lgn_fr_vector,lgn_error_vector,mode='norm')
    # plot_time_contrast(lgn_fr_vector,lgn_error_vector,mode='mean')
    # plot_time_contrast(lgn_fr_vector,lgn_error_vector,mode='max')
    # Plot_Log_Log_fig('lgn')
    # Plot_Log_Log_fig('v1_fr')
    # Plot_Log_Log_fig('v1_error')

    

