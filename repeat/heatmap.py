
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import os
from repeat_dw import read_output
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示
import warnings
warnings.filterwarnings('ignore')

#读取所有time所有contrast重复10次的数据，输出：(5,10,10,3840)
def merge_data():
	fr_list=[]
	for time in times:
		for k in range(n_pic):
			for i in range(repeat):
				fr_vector=read_output(dir+'/contrast{}/fr_time{}.npz'.format(k+1,time))
				fr_list.append(fr_vector[i])
	fr_vector = np.array(fr_list).reshape(len(times),n_pic, repeat, -1)
	return fr_vector 
 
#对不同时间时，重复十次fr及其误差的热图fr_vector为上一个函数读出来的数据（5，10，10，3840）
#画两张4x5的图
def plot_fr_error(fr_vector):
	for time in times:
		fr_tensor = fr_vector[time-1] #(10,10,3840)
		fr_mean = np.mean(fr_tensor,axis=1)
		#图1：都以fr_mean上升排序
		fig = plt.figure(figsize=(60,40),dpi=200)
		for i in range(10):#n_pic):
			ax = fig.add_subplot(4,5,i+1)
			fr_mean_df=pd.DataFrame(fr_mean[i]).T.round(2)
			fr_df = pd.DataFrame(fr_tensor[i])#(10,3840)
			df1=fr_mean_df.append(fr_df, ignore_index=True)
			df1=df1.sort_values(by=0,axis=1,ascending=True)
			df1.columns = df1.loc[0]
			df1 = df1[1:]
			sns.heatmap(data=df1,
						cmap=plt.get_cmap('Greens'),#matplotlib中的颜色盘'Greens'
						vmin=0,#图例（右侧颜色条color bar）中最小显示值 
						vmax=60,#图例（右侧颜色条color bar）中最大显示值
					)
			plt.title("fr:contrast={}".format((i+1)/20),fontsize = 20)

			ax = fig.add_subplot(4,5,i+11)
			error_df=(fr_df-fr_df.iloc[0]).abs()
			df2=fr_mean_df.append(error_df, ignore_index=True)
			df2=df2.sort_values(by=0,axis=1,ascending=True)
			df2.columns = df2.loc[0]
			df2 = df2[1:]
			sns.heatmap(data=df2,
						cmap=plt.get_cmap('Greens'),
						vmin=0,
						vmax=2,
					)
			plt.title("error:contrast={}".format((i+1)/20),fontsize = 20)
		plt.suptitle('heatmap(sorted by mean fr,time={}s)'.format(time), ha = 'left',fontsize = 30, weight = 'extra bold')
		plt.savefig(os.path.join(dir+'/merged', 'heatmap(sorted by mean frtime={}s)'.format(time)))#第一个是指存储路径，第二个是图片名字
		plt.close()
		
		#图2：fr以fr_mean上升排序,error以error_mean排序
		fig = plt.figure(figsize=(60,40),dpi=200)
		for i in range(10):#n_pic):
			ax = fig.add_subplot(4,5,i+1)
			fr_mean_df=pd.DataFrame(fr_mean[i]).T.round(2)
			fr_df = pd.DataFrame(fr_tensor[i])#(10,3840)
			df1=fr_mean_df.append(fr_df, ignore_index=True)
			df1=df1.sort_values(by=0,axis=1,ascending=True)
			df1.columns = df1.loc[0]
			df1 = df1[1:]
			sns.heatmap(data=df1,
						cmap=plt.get_cmap('Greens'),
						vmax=60,
					)
			plt.title("fr:contrast={}".format((i+1)/20),fontsize = 20)

			ax = fig.add_subplot(4,5,i+11)
			error_df=(fr_df-fr_df.iloc[0]).abs().round(2)
			error_df.loc[10] = error_df.apply(lambda x: x.mean(),axis=0).round(2)
			df2=error_df.sort_values(by=10,axis=1,ascending=True)
			df2.columns = df2.loc[10]
			df2 = df2[:10]
			sns.heatmap(data=df2,
						cmap=plt.get_cmap('Greens'),
						vmin=0,
						vmax=2,
					)
			plt.title("error:contrast={}".format((i+1)/20),fontsize = 20)
		plt.suptitle('heatmap(sorted by mean error,time={}s)'.format(time), ha = 'left',fontsize = 30, weight = 'extra bold')
		plt.savefig(os.path.join(dir+'/merged', 'heatmap(sorted by mean error,time={}s)'.format(time)))#第一个是指存储路径，第二个是图片名字
		plt.close()


if __name__ == "__main__" :
	times=[1,2,3,4,5]  #运行时间
	n_pic=10      #图片个数
	repeat=10    #重复次数
	dir='/home/zhaobenyan/dataset/output/driftgrating_32x32'    #需要读取的数据所在路径
	fr_vector=merge_data()
	plot_fr_error(fr_vector)
