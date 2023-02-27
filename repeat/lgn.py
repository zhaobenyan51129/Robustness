import matplotlib.pyplot as plt
import numpy as np
import os
from repeat_dw import read_output
np.set_printoptions(threshold=np.inf)  #使输出数据完整显示

time=1
n_pic=10      #图片个数
repeat=10    #重复次数
dir='/home/zhaobenyan/dataset/output/driftgrating_32x32'    #需要读取的数据所在路径
lgn=read_output(dir+'/contrast5/lgn_time{}.npz'.format(time))
print(np.array(lgn).shape)
ar,num=np.unique(lgn[9],return_counts=True)
print(ar)
#print([num[0],np.sum(num[1:])])
print(num)
print(512*8000)

explode = (0, 0.1)  # only "explode" the 1st slice (i.e. '0')
numbers=[num[0],np.sum(num[1:])]
labels=['zero','nonzero']
fig1, ax1 = plt.subplots()
ax1.pie(numbers, labels=labels,explode=explode, autopct='%1.11f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig('lgn_spike.jpg')
plt.show()