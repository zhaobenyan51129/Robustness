import numpy as np

file_name='/home/zhaobenyan/data/dw_test_new/Repeatability_contrast/Contrast1/output_vector.npz'
with np.load(file_name) as f:
      sampleID=f['sampleID']
      unattacked_output_v1=f['unattacked_output_v1'] 
      attack_v1=f['attack_v1']
      error_vector_v1=f['error_vector_v1']
print(unattacked_output_v1.shape)