import torch
import numpy as np

def daiwei_order(image_size):# 注意image_size需为偶数
    indices_list = []
    a = image_size/2 - 1
    b = image_size/2
    for i in range(image_size):
        for j in range(image_size):
            if (i-a)**2+(j-a)**2<=b**2 or (i-a)**2+(j-b)**2<=b**2 or (i-b)**2+(j-a)**2<=b**2 or (i-b)**2+(j-b)**2<=b**2:
                indices_list.append(i * image_size + j)
    indices_tensor_channel0 = torch.tensor(indices_list)
    indices_tensor_channel1 = indices_tensor_channel0 + image_size**2
    indices = torch.cat([indices_tensor_channel0, indices_tensor_channel1], dim=0)
    shuffle_idx = torch.randperm(indices.numel())
    indices = indices[shuffle_idx]
    return indices


def twice_daiwei_order(image_size):# 注意image_size需为偶数
    indices_list = []
    a = image_size/2 - 1
    b = image_size/2
    for i in range(image_size):
        for j in range(image_size):
            if (i-a)**2+(j-a)**2<=b**2 or (i-a)**2+(j-b)**2<=b**2 or (i-b)**2+(j-a)**2<=b**2 or (i-b)**2+(j-b)**2<=b**2:
                indices_list.append(i * image_size + j)
    indices_tensor_channel0 = torch.tensor(indices_list)
    indices_tensor_channel1 = indices_tensor_channel0 + image_size**2
    indices = torch.cat([indices_tensor_channel0, indices_tensor_channel1], dim=0)
    shuffle_idx1 = torch.randperm(indices.numel())
    shuffle_idx2 = torch.randperm(indices.numel())
    indices1 = indices[shuffle_idx1]
    indices2 = indices[shuffle_idx2]

    return torch.cat([indices1, indices2], dim=0)

# def ljy_bin_image_to_tensor(file_dir):
#     with open(file_dir) as f:
#         x_1 = np.fromfile(f, 'i4', 1)
#         x_2 = np.fromfile(f, 'i4', 3)
#         x_3 = np.fromfile(f, 'f4', 3)
#         x_4 = np.fromfile(f, 'f4', 2)
#         x_5 = np.fromfile(f, 'u4', 1)
#         x_6 = np.fromfile(f, 'f4', 16*16*3)
#         image_numpy = x_6.reshape(3, 16, 16)
#         image_tensor = torch.tensor(image_numpy)   
#     return image_tensor # [3, 16, 16]

def ljy_bin_image_to_tensor(file_dir):
    with open(file_dir) as f:
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

if __name__ == "__main__":
    print(twice_daiwei_order(8)) 