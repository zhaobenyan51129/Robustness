import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def f(x, y, a):
    return a * np.log(x / y)

def hat_f(x, y, a, b):
    return a * np.log(x / y) + b * np.sign(np.log(x / y))

def reverse_f(x,y,a):
    return x * np.exp(-y / a)


# 定义画图函数
def plot_illusion(grey_low,grey_high,grey_middle,width,locate,a,b,pic_name):
    '''
    注：灰度值越大，颜色越浅。黑色表示灰度值为0，白色表示灰度值为1。
    grey_low: 左图的背景灰度值，范围在0到1之间，数值越小，背景颜色越浅
    grey_high: 右图的背景灰度值，范围在0到1之间，数值越小，背景颜色越浅
    grey_middle: 中间方块的灰度值，范围在0到1之间，数值越小，方块越浅
    width: 方块的宽度
    locate: 方块左上角在图像中的位置，为二元组(x, y)
    '''
    img_save_path='/home/zhaobenyan/dataset/data_ill'
    gray_square = np.ones((width, width, 3)) * grey_middle
    gray_low_bg = np.ones((300, 300, 3)) * grey_low
    gray_low_bg[locate[0]:locate[0]+width, locate[1]:locate[1]+width, :] = gray_square
    gray_high_bg = np.ones((300, 300, 3)) * grey_high
    gray_high_bg[locate[0]:locate[0]+width, locate[1]:locate[1]+width, :] = gray_square
    gray=np.ones((300, 300, 3)) * grey_middle

    y_left_hat=hat_f(grey_low,grey_middle,a,b)
    y_right_hat=hat_f(grey_high,grey_middle,a,b)
    hat_left=reverse_f(grey_low,y_left_hat,a)
    hat_right=reverse_f(grey_high,y_right_hat,a)
    hat_left_bg = np.ones((300, 300, 3)) * hat_left
    hat_right_bg = np.ones((300, 300, 3)) * hat_right

    fig, axs = plt.subplots(2, 3, figsize=(8, 4))
    axs[0, 0].imshow(gray_low_bg)
    axs[0, 0].set_title(f'background={grey_low}')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(gray_high_bg)
    axs[0, 1].set_title(f'background={grey_high}')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(gray)
    axs[0, 2].set_title(f'grey={grey_middle}')
    axs[0, 2].axis('off')
    axs[1, 0].imshow(hat_left_bg)
    axs[1, 0].set_title(f'hat_left_bg={np.round(hat_left,6)}')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(hat_right_bg)
    axs[1, 1].set_title(f'hat_right_bg={np.round(hat_right,6)}')
    axs[1, 1].axis('off')
    axs[1, 2].imshow(gray)
    axs[1, 2].set_title(f'grey={grey_middle}')
    axs[1, 2].axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # fig.suptitle(f"a={a},b={b}", fontsize=16)
    plt.savefig(os.path.join(img_save_path, f'{pic_name}'))#第一个是指存储路径，第二个是图片名字
    plt.close()

def plot_illusions(inputs):
    for i, (grey_low,grey_high,grey_middle,width,locate,a,b) in enumerate(inputs):
        plot_illusion(grey_low,grey_high,grey_middle,width,locate,a,b,i)
    
def images_to_video(img_path, video_name):
    img_array = []
    imgList = os.listdir(img_path)
    imgList.sort(key=lambda x: float(x.split('.')[0]))  
    # print(f"imgList={imgList}")

    for count in range(len(imgList)): 
        filename = imgList[count]
        # print(f"filename={filename}")
        img = cv2.imread(os.path.join(img_path, filename))
        # print(f"img={img}")
        if img is None:
            print(filename + " is an error!")
            continue
        img_array.append(img)

    height, width, layers = img_array[0].shape
    size = (width, height)
    fps = 20  # 设置每帧图像切换的速度
    output_path = f'/home/zhaobenyan/dataset/data_videos/{video_name}.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == "__main__" :
    inputs = [(0.8, 0.2, gray, 200, (50, 50),0.35,0.003) for gray in list(np.round(np.linspace(0.01, 1, 201),2))[::-1]]
    plot_illusions(inputs)
    img_save_path='/home/zhaobenyan/dataset/data_ill'
    images_to_video(img_save_path,'change_middle')
print('-----------finished------------')  