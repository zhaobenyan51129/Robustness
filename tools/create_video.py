import cv2
import os

def images_to_video(path):
    img_array = []
    
    imgList = os.listdir(path)
    #imgList.sort(key=lambda x: int(x.replace("layer","").split('.')[0])) 
    imgList.sort(key=lambda x: float(x.split('.')[0]))  
    #print(imgList)
    for count in range(0, len(imgList)): 
        filename = os.path.join(imgList[count])
        img = cv2.imread(path + filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    height, width, layers = img.shape
    size = (width, height)
    fps = 5  # 设置每帧图像切换的速度
    out = cv2.VideoWriter('/home/zhaobenyan/data/data2/vit_curve/attack100_vit.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
 
def main():
    path = "/home/zhaobenyan/data/data2/evolution1/"  # 改成你自己图片文件夹的路径
    images_to_video(path)
 
if __name__ == "__main__":
    main()