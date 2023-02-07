import numpy as np
import matplotlib.pyplot as plt
import torch


#生成gauss噪声
def gauss_noise(size):
    np.random.seed(167234352)
    normal_noise=np.random.normal(loc=0, scale=1, size=[size, size, 3])
    torch.save(normal_noise, '/home/zhaobenyan/Robustness_test/files/normal_noise_{}.pth'.format(size))

#生成均匀分布噪声
def uniform_noise(size):
    np.random.seed(167234352)
    uniform_noise=np.random.uniform(low=0, high=1, size=[size, size, 3])
    torch.save(uniform_noise, '/home/zhaobenyan/Robustness_test/files/uniform_noise_{}.pth'.format(size))

#读取噪声数据 file:噪声.pth文件所在地址
def read_noise(file):
    noise = torch.load(file)
    size=noise.shape[0]
    plt.figure(figsize=(3, 3))
    plt.imshow(noise)
    plt.colorbar()
    plt.savefig('/home/zhaobenyan/Robustness_test/files/normal_noise_{}.jpg'.format(size))
    plt.close()

gauss_noise(256)
read_noise('/home/zhaobenyan/Robustness_test/files/normal_noise_256.pth')