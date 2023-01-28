import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

def read_data(file_name):
      '''
      sampleID (3084,)
      unattacked_output_v1 (3084,)
      attack_v1 (3, 3084)
      error_vector_v1 (3, 3084)
      '''
      with np.load(file_name) as f:
            sampleID=f['sampleID']
            unattacked_output_v1=f['unattacked_output_v1'] 
            attack_v1=f['attack_v1']
            error_vector_v1=f['error_vector_v1']
      return sampleID, unattacked_output_v1, attack_v1,error_vector_v1  

file='/home/zhaobenyan/data/dw_test_new/Repeatability_contrast/Contrast6/output_vector.npz' 
sampleID, unattacked_output_v1, attack_v1,error_vector_v1=read_data(file)
 
# a=[[1,2,3,4],[5,6,7,8]]       #包含两个不同的子列表[1,2,3,4]和[5,6,7,8]
all_vercor=[]
all_vercor.append(sampleID)
all_vercor.append(unattacked_output_v1)
for i in range(3):
      all_vercor.append(attack_v1[i])
for i in range(3):
      all_vercor.append(error_vector_v1[i])


data=DataFrame(all_vercor)#这时候是以行为标准写入的
data=data.T#转置之后得到想要的结果
data.rename(columns={0:'sampleID',1:'unattacked_output',2:'attack_1st',3:'attack_2nd',4:'attack_3rd',5:'error_1st',6:'error_2nd',7:'error_3rd'},inplace=True)#注意这里0和1都不是字符串
data['samleID_error!=0_any'] =data.apply(lambda x: x['sampleID'] if x['error_1st'] !=0 or x['error_2nd']!=0 or x['error_3rd']!=0 else np.nan, axis=1)
data['samleID_error!=0_all'] =data.apply(lambda x: x['sampleID'] if x['error_1st'] !=0 and x['error_2nd']!=0 and x['error_3rd']!=0 else np.nan, axis=1)
print(data.info())