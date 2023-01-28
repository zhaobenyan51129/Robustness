import numpy as np

path='/home/zhaobenyan/data/resource_contrast32_new/'
#path='/home/zhaobenyan/data/resource_grid_benchmark_224pixel/'
with open(path+'static_color-grid_6_cfg.bin') as f: #可以去查查with是干嘛的，它是以防你忘记关掉文件，这种表达会比较好
    sf = np.fromfile(f,'f4',1)[0] #为啥是np.fromfile（。。。）[0],因为用np命令的读出来是array，其实我们只是要他的元素
    ori = np.fromfile(f,'f4',1)[0]
    phase = np.fromfile(f,'f4',1)[0]
    contrast = np.fromfile(f,'f4',1)[0] #我一般读取到contrast就停了
    #crest = np.fromfile(f,'f4',3)  #波峰
    #valley = np.fromfile(f,'f4',3) #波谷
print('spatial frequency:',sf)
print('oritation:',ori)
print('phase',phase)
print('contrast',contrast)
