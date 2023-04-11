import numpy as np
import matplotlib.pyplot as plt
import shutil
import sys
from io import StringIO
import threading
import time as Time
import os
import queue
import fileinput
output_buffer = StringIO()
old_stdout = sys.stdout
sys.stdout = output_buffer
BUILD_UP_DIR='/home/zhaobenyan/buildup/'
sys.path.insert(0,f'{BUILD_UP_DIR}')
import utils
from generate import *
sys.stdout = old_stdout


#通过import *的模式，所有用到的库就全import了，不需要再import了。AKA，只要一次import
PATCHFAST_DIR='/home/zhaobenyan/model/patchfast/'
RESOURCE_DIR='/home/zhaobenyan/model/resource/'
SRC_DIR='/home/zhaobenyan/model/repos/patchV1/src/'
BIN_DIR='/home/zhaobenyan/model/bin/'
MAX_RESOURCE=32 #最大并行数量，一般来说设置为显卡数量的整数倍，已经经过调优。注意：您仍可以并行超过这个数量，并且这样做是受到推荐的，因为它将更快。只不过超过Max后会有一定的串行情况。 
VERSION='v1.2'
#以上，通过buildup进行相应设定，改动请重新buildup

#以下，可以自行设定
data_save_path='/home/zhaobenyan/model/data_save/auto_manage/' #自己设定，数据保存文件夹 以‘/’结尾 
img_save_path='/home/zhaobenyan/model/data_save/img_save/' #自己设定，是Usave_img的根目录 
model_log_path='/home/zhaobenyan/model/data_save/log/' #自己设定,存放戴老师模型的输出，可以通过观察是否有一个到100%的进度条来判断模型运行是否正常 
gratings_path='/home/zhaobenyan/model/data_save/gratings/' #generate.py里面生成grating的路径

sleep_time=0.01 #线程连续启动时的等待时间参数，调参时使用，现在我应该调好了（如果时间间隔过短，则patch_fast会都选中同一GPU导致计算资源浪费）
log_on=False #开启log方便调试

#此版本模型，返回两个值，第一个是512个lgn，在每个时间段上的发放次数，shape:(512, 8000*秒数) 
#                      第二个是v1各神经元的放电率，shape:(,3840)
#使用参数 layer='lgn'获得第一个返回值，使用layer='v1'或不写，获得第二个返回值
#不用担心会重复计算

for directory in [data_save_path,img_save_path,model_log_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def _add_path(new_path):
    # 获取当前环境变量
    env = os.environ.copy()
    # 如果环境变量中不存在该路径，将其添加到PATH变量中
    if 'PATH' in env:
        paths = env['PATH'].split(':')
        if new_path not in paths:
            env['PATH'] = f"{new_path}:{env['PATH']}"
    else:
        env['PATH'] = new_path
    # 更新环境变量
    os.environ.update(env)

_add_path(f'{BIN_DIR}')
_add_path(f"{SRC_DIR}util/")


def Upeek_target_neurons(experiments, target_neurons, save_path=""):
    """
    绘制目标神经元在每个实验中的响应。

    参数：
    experiments: numpy array, shape (n_experiments, n_neurons)
        包含实验结果的矩阵，每行是一个实验的神经元响应。
    target_neurons: list
        包含关注的神经元编号的列表。

    返回：
    None
    """

    # 获取目标神经元的响应
    target_responses = experiments[:, target_neurons]

    # 绘制条形图
    n_experiments = target_responses.shape[0]
    n_target_neurons = target_responses.shape[1]
    ind = np.arange(n_experiments)  # x轴坐标
    width = 0.8 / n_target_neurons  # 条形图宽度
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 颜色列表

    fig, ax = plt.subplots()
    rects = []
    for i in range(n_target_neurons):
        color = colors[i % len(colors)]  # 循环使用颜色
        rect = ax.bar(ind + (i - n_target_neurons/2 + 0.5) * width, target_responses[:, i], width) #, color=color
        rects.append(rect)

    # 添加标签、标题和图例
    ax.set_ylabel('Response')
    ax.set_title('Target Neuron Responses')
    ax.set_xticks(ind)
    ax.legend(rects, [f'Neuron {n}' for n in target_neurons])

    if save_path!="":
        plt.savefig(save_path)
    plt.show()

def _generate_Name_by_Para(C, P, SF, D, frameRate, size, repeat_time=1,run_time=0):
    return f"C{C:.3f}_P{P:.3f}_SF{SF}_D{D:.3f}_frameRate{frameRate}_size{size}_Rpt{repeat_time}_RunT{run_time:.3f}"

def Uget_Image_Array(C, P, SF, D, frameRate, size):
    """C, P, SF, D, frameRate, size, --彩票收到烦死 --菜品速度奉上"""
    path=img_save_path+_generate_Name_by_Para(C, P, SF, D, frameRate, size)+now_time_str()+'/'
    mk_path(path)
    sys.stdout = output_buffer
    generate_input(path, np.array([C]), np.array([P]), np.array([SF]), np.array([D]), frameRate, size)
    image_path=path+"static_color-grid_1.bin"
    image_array=utils.ljy_bin_image_to_tensor(image_path)
    sys.stdout = old_stdout
    shutil.rmtree(path)
    return image_array

#读取输出LGN_spike_time:[521,8000], fr:[3840,1],
def read_spike(file):
    with open(file) as f:
        sampleSize = np.fromfile(f, 'u4', 1)[0] #一共5120
        sample_t0, sample_t1 = np.fromfile(f, 'f4', 2)#t0是开始时间，t1是结束时间，我一共跑了1s
        nt = np.fromfile(f, 'u4', 1)[0]
        nLGN = np.fromfile(f, 'u4', 1)[0]
        LGN_spike_time = np.fromfile(f, 'f4', nLGN*nt)
        sampleID = np.fromfile(f, 'u4', sampleSize)#id排序是顺序的
        sample_spikeCount = np.fromfile(f, 'u4', sampleSize)
        fr = sample_spikeCount/(sample_t1-sample_t0)*1000
    LGN_spike_time = LGN_spike_time.reshape((nt,nLGN)).T
    return LGN_spike_time,fr

def run_model(resource_queue,res_dic,res_tag,run_time=1,image=None,x1=None,x2=None,x3=None,x4=None,x5=None):
    this_success=False
    if log_on:
        print("run_model") #LOG
    while (this_success==False):
        try:
            resource_index = resource_queue.get(block=False)
        except queue.Empty:
            # 如果队列为空，则等待
            Time.sleep(0.015*(1+3*np.random.rand()))
        else:
            # 执行读写操作
            print(f"get_{resource_index}")
            this_success=True
            with open(f"{SRC_DIR}/minimal{resource_index}.cfg", 'r') as file:
                # 读取文件的每一行，并用列表保存
                lines = file.readlines()
            for i in range(len(lines)):
                if "nt =" in lines[i] and "#nt =" not in lines[i] and lines[i][0]=='n':
                    lines[i] = f"nt ={int(8000*run_time)} \n"
                    if log_on:
                        print(f"changed {int(8000*run_time)}") #LOG
            with open(f"{SRC_DIR}/minimal{resource_index}.cfg", 'w') as file:
                # 将修改后的行重新写入到文件中
                file.writelines(lines)

            with open(f"{RESOURCE_DIR}static_color-grid_1{resource_index}.bin",'wb') as f:
                if len(image.shape)==4:
                    image = image[0]  
                    imgsize=image.shape[0]  
                np.array(x1).astype('i4').tofile(f) 
                np.array(x2, dtype='i4').tofile(f)
                np.array(x3).astype('f4').tofile(f) # init_luminance
                np.array(x4, dtype='f4').tofile(f)
                np.array(x5).astype('u4').tofile(f)
                y = image.transpose([2,0,1]).flatten()
                y.astype('f4').tofile(f)
            os.system(f'cd {PATCHFAST_DIR} && {BIN_DIR}patch_fast -c {SRC_DIR}minimal{resource_index}.cfg > {model_log_path}{now_time_str()}.log') 
            LGN_spike_time,fr=read_spike(f"{PATCHFAST_DIR}sample_spikeCount-merge_test_1{resource_index}.bin")
            res_dic[res_tag].append([LGN_spike_time,fr])
            # 将资源文件名称放回队列中
            resource_queue.put(resource_index)
            if log_on:
                print(f"back_{resource_index}") #LOG
            return
            
                
        
            
# image,repeat_times=1,x1=None,x2=None,x3=None,x4=None,x5=None,size=32,frameRate=1,run_time=1
def Uget_Image_Response(input_list):
    """
    使用指定的图像作为输入，获得神经网络的响应值。
    [repeat_times,res_tag,run_time,image,x1,x2,x3,x4,x5]
    参数：
    image: np.ndarray
        要处理的输入图像，必须是3通道（BGR）的numpy数组，形状为(H,W,3)。
    repeat_times: int, 默认值为1
        重复执行模型的次数。可以提高响应值的准确性，但会增加计算时间。
    x1: list, 默认值为None
        一个包含float值的列表，表示模型的第1个输入张量的值。列表长度必须是1。如果x1为None，则默认为[-1]。
    x2: list, 默认值为None
        一个包含float值的列表，表示模型的第2个输入张量的值。列表长度必须是3。如果x2为None，则默认为[frameRate/8, size, size]。
    x3: list, 默认值为None
        一个包含float值的列表，表示模型的第3个输入张量的值。列表长度必须是3。如果x3为None，则默认为[0.5, 0.5, 0.5]。
    x4: list, 默认值为None
        一个包含float值的列表，表示模型的第4个输入张量的值。列表长度必须是2。如果x4为None，则默认为[0.05110313, 0.0344]。
    x5: list, 默认值为None
        一个包含float值的列表，表示模型的第5个输入张量的值。列表长度必须是1。如果x5为None，则默认为[1]。
    size: int, 默认值为32
        输入图像的大小（较短的边长）。默认值为32。
    frameRate: int, 默认值为1
        图像序列的帧率。默认值为1。
    run_time: float
        运行模型的时间，默认为1s。

    返回：
    mean_result_lgn: np.ndarray
        平均的神经网络响应值，形状为(512, 8000*秒数)。
    mean_result: np.ndarray
        平均的神经网络响应值，形状为(, 3840)。
    """
    # print(f"repeat:{repeat_times}")

    resource_queue=queue.Queue()
    for i in range(MAX_RESOURCE):
        if log_on:
            print(f"put_{i}") #LOG
        resource_queue.put(i)
    #返回值字典
    res_dic={}
    #所有线程
    threads = []
    
    for input_args in input_list:
        repeat_times=input_args[0] #TODO:改
        res_tag=input_args[1]
        run_time=input_args[2]
        image=input_args[3]
        x1=input_args[4]
        x2=input_args[5]
        x3=input_args[6]
        x4=input_args[7]
        x5=input_args[8]

        res_dic[res_tag]=[]

        for i in range(repeat_times):
            thread = threading.Thread(target=run_model, args=(resource_queue,res_dic,res_tag,run_time,image,x1,x2,x3,x4,x5))
            threads.append(thread)
        
    # 开始执行线程
    for thread in threads:
        thread.start()
        Time.sleep(sleep_time)

    # 等待所有线程执行完毕
    for thread in threads:
        thread.join()

    for key, value in res_dic.items():
        LGN_mean = np.mean([res[0] for res in value],axis=0)
        v1_mean = np.mean([res[1] for res in value],axis=0)
        if log_on:
            print(f"LGN shape{LGN_mean.shape}") #LOG
            print(f"v1 shape{v1_mean.shape}") #LOG
        res_dic[key]=[LGN_mean,v1_mean]
        
    # output_str = output_buffer.getvalue()
    # with open(f'{model_log_path}{now_time_str()}.log', 'w') as f:
    #     f.write(output_str)
    return res_dic



def Uget_Parameter_Response(C, P, SF, D, frameRate, size ,repeat_times=1,run_time=1,test_mod=False):
    """
    根据参数生成图像，并获取图像响应。

    Args:
        - C: float，图像对比度，范围为[0, 0.5]。
        - P: float，图像相位，范围为[0, pi]。
        - SF: float，图像空间频率。
        - D: float，图像方向，单位为弧度，范围为[0, pi]。
        - frameRate: float，帧率，单位为 Hz，取值为1, 72。
        - size: int，图像大小，单位为像素。
        - repeat_times (int, optional): 重复次数。默认为1。
        - run_time (float, optional): 每个重复的运行时间，单位秒。默认为1。
        - test_mod (bool, optional): 是否在测试模式下运行。默认为False。

    Returns:
        [LGN,v1]: 图像响应。 
            LGN: (512,8000*run_time)
            v1: (3840,)


    Example:
        C = 0.4
        P = 0
        SF = 24
        D = 0
        frameRate = 1
        size = 32
        res = Uget_Parameter_Response(C, P, SF, D, frameRate, size)

    注意事项：
    如果目标数据已经存在，则会直接从本地文件系统中加载已保存的响应结果。
    如果目标数据不存在，则会计算响应并将结果保存到本地文件系统中。
    """
    res_tag=_generate_Name_by_Para(C, P, SF, D, frameRate, size,repeat_times,run_time)
    res_path=data_save_path+res_tag+".npy"
    if os.path.isfile(res_path) and not test_mod:
        if log_on:
            print("loaded") #LOG
        return np.load(res_path,allow_pickle=True)
    else:
        image=Uget_Image_Array(C, P, SF, D, frameRate, size)
        # [repeat_times,res_tag,run_time,image,x1,x2,x3,x4,x5]
        x1=[-1] #固定是-1
        x2=[frameRate/8,size,size] #frameRate/8，size,size
        x3=[0.5, 0.5, 0.5] #固定是0.5*3
        x4=[0.05110313, 0.0344] #模型设置好之后的固定值
        x5=[1] #一个眼睛，现在只能填1
        res_dic=Uget_Image_Response([[repeat_times,res_tag,run_time,image,x1,x2,x3,x4,x5]])
        if not test_mod:
            np.save(res_path,res_dic[res_tag])
        return res_dic[res_tag]
    
def Uget_Parameter_Response_parallel(arg_list,test_mod=False):
    """
    使用多线程计算多张图像的神经元响应，并保存响应结果。

    Args:
    - arg_list: list，包含多个参数列表的列表，每个参数列表对应一张图像的计算，格式为 [C, P, SF, D, frameRate, size, repeat_times, layer]。
        - C: float，图像对比度，范围为[0, 0.5]。
        - P: float，图像相位，范围为[0, pi]。
        - SF: float，图像空间频率。
        - D: float，图像方向，单位为弧度，范围为[0, pi]。
        - frameRate: float，帧率，单位为 Hz，取值为1, 72。
        - size: int，图像大小，单位为像素。
        - repeat_times: int，每张图像计算的重复次数，默认为1。

    - test_mod: bool，是否以测试模式运行，测试模式下会跳过已保存的结果，只返回新计算的结果，默认为False。

    Returns:
    - res_list: list，包含多个响应结果的列表，每个响应结果对应一张图像的计算。
        [LGN,v1]: 图像响应。 
            LGN: (512,8000*run_time)
            v1: (3840,)

    Examples:
    >>> arg_list = [[0.5, 0, 24, np.pi/2, 1, 32], [0.4, pi, 24, 0, 1, 32]] （有问题）
    >>> res_list = Uget_Parameter_Response_parallel(arg_list)
    """

    input_list=[]
    input_list_len=len(arg_list)
    res_list = [None] * input_list_len
    res_tag_list=[""]*input_list_len
    res_path_list=[""]*input_list_len
    for i in range(input_list_len):
        list=arg_list[i]
        C=list[0]
        P=list[1]
        SF=list[2]
        D=list[3]
        frameRate=list[4]
        size=list[5]
        try:
            repeat_times=list[6]  # 尝试取出第7个元素
        except IndexError:
            repeat_times=1  # 出现异常时返回默认值
        try:
            run_time=list[7]  # 尝试取出第8个元素
        except IndexError:
            run_time=1  # 出现异常时返回默认值        
        
        res_tag_list[i]=_generate_Name_by_Para(C, P, SF, D, frameRate, size,repeat_times,run_time)
        res_path_list[i]=data_save_path+res_tag_list[i]+".npy"
        if os.path.isfile(res_path_list[i]) and not test_mod:
            res_list[i]=np.load(res_path_list[i],allow_pickle=True)
        else:
            image=Uget_Image_Array(C, P, SF, D, frameRate, size)
            # [repeat_times,res_tag,run_time,image,x1,x2,x3,x4,x5]
            x1=[-1] #固定是-1
            x2=[frameRate/8,size,size] #frameRate/8，size,size
            x3=[0.5, 0.5, 0.5] #固定是0.5*3
            x4=[0.05110313, 0.0344] #模型设置好之后的固定值
            x5=[1] #一个眼睛，现在只能怪填1
            input_list.append([repeat_times,res_tag_list[i],run_time,image,x1,x2,x3,x4,x5])

    res_dic=Uget_Image_Response(input_list)
    for i in range(input_list_len):
        if res_list[i] == None:
            res_list[i]=res_dic[res_tag_list[i]]
            if not test_mod:
                np.save(res_path_list[i],res_list[i])
    return res_list


def Uget_Image_Response_with_SaveTag(image,save_tag,size=32,frameRate=1,repeat_times=1,run_time=1,test_mod=False):
    """
    获取给定图像的神经元响应，并将响应结果保存到本地文件系统中。

    参数：
    image: numpy数组，表示输入的图像。通常情况下，这个数组的形状应该是 (height, width, channels)。
    save_tag: str，表示保存文件的标签。文件将保存在 data_save_path 目录下，文件名将根据标签和其他参数自动生成。
    size: int，可选参数，表示图像的缩放大小。默认值为 32。
    frameRate: int，可选参数，表示重复图像的帧速率。默认值为 1。
    repeat_times: int，可选参数，表示获取响应的重复次数。默认值为 1。
    run_time: int，可选参数，表示获取响应的运行时间。默认值为 1。
    test_mod: bool，可选参数，表示是否为测试模式。默认为 False。

    Returns:
        [LGN,v1]: 图像响应。 
            LGN: (512,8000*run_time)
            v1: (3840,)

    注意事项：
    如果指定的标签已经存在，则会直接从本地文件系统中加载已保存的响应结果。
    如果指定的标签不存在，则会计算响应并将结果保存到本地文件系统中。
    """
    res_tag=save_tag+f"_Rpt{repeat_times}_RunT{run_time}"
    res_path=data_save_path+res_tag+".npy"
    if os.path.isfile(res_path) and not test_mod:
        if log_on:
            print("loaded") #LOG
        return np.load(res_path,allow_pickle=True)
    else:
        # [repeat_times,res_tag,run_time,image,x1,x2,x3,x4,x5]
        x1=[-1] #固定是-1
        x2=[frameRate/8,size,size] #frameRate/8，size,size
        x3=[0.5, 0.5, 0.5] #固定是0.5*3
        x4=[0.05110313, 0.0344] #模型设置好之后的固定值
        x5=[1] #一个眼睛，现在只能怪填1

        res_dic=Uget_Image_Response([[repeat_times,res_tag,run_time,image,x1,x2,x3,x4,x5]])
        if not test_mod:
            np.save(res_path,res_dic[res_tag])
        return res_dic[res_tag]
    
def Uget_Image_Response_with_SaveTag_parallel(arg_list, test_mod=False):
    """
    使用多线程计算多张图像的神经元响应，并保存响应结果。

    参数：
    arg_list : list
        包含多个参数列表的列表，每个参数列表对应一张图像的计算。参数列表的格式为[image, save_tag, repeat_times=1, layer="v1", size=32, frameRate=1]。
            image : numpy.ndarray
                输入图像数组。(height, width, channels)
            save_tag : str
                保存标签。
            repeat_times : int, optional
                重复次数，默认为1。
            layer : str, optional
                获取的神经元层级，默认为"v1"。
            size : int, optional
                图像尺寸，默认为32。
            frameRate : int, optional
                帧率，默认为1。

    返回：
    list
        包含多个响应结果的列表，每个响应结果对应一张图像的计算。
        [LGN,v1]: 图像响应。 
            LGN: (512,8000*run_time)
            v1: (3840,)

    例子：
    >>> image1 = np.random.rand(32, 32, 3)
    >>> image2 = np.random.rand(32, 32, 3)
    >>> arg_list = [[image1, "image1_result"], [image2, "image2_result"]]
    >>> results = Uget_Image_Response_with_SaveTag_parallel(arg_list)
    """
    input_list=[]
    input_list_len=len(arg_list)
    res_list = [None] * input_list_len
    res_tag_list=[""]*input_list_len
    res_path_list=[""]*input_list_len
    for i in range(input_list_len):
        # image,save_tag,size=32,frameRate=1,repeat_times=1,run_time=1,test_mod=False
        list=arg_list[i]
        image=list[0]
        save_tag=list[1]
        
        try:
            size=list[2] # 尝试取出第3个元素
        except IndexError:
            size=32  # 出现异常时返回默认值
        try:
            frameRate=list[3]  # 尝试取出第4个元素
        except IndexError:
            frameRate=1  # 出现异常时返回默认值       
        try:
            repeat_times=list[4] # 尝试取出第5个元素
        except IndexError:
            repeat_times=1  # 出现异常时返回默认值
        try:
            run_time=list[5]  # 尝试取出第6个元素
        except IndexError:
            run_time=1  # 出现异常时返回默认值      

        res_tag_list[i]=save_tag+f"_Rpt{repeat_times}_RunT{run_time}"
        res_path_list[i]=data_save_path+res_tag_list[i]+".npy"
        if os.path.isfile(res_path_list[i]) and not test_mod:
            res_list[i]=np.load(res_path_list[i],allow_pickle=True)
        else:
            # [repeat_times,res_tag,run_time,image,x1,x2,x3,x4,x5]
            x1=[-1] #固定是-1
            x2=[frameRate/8,size,size] #frameRate/8，size,size
            x3=[0.5, 0.5, 0.5] #固定是0.5*3
            x4=[0.05110313, 0.0344] #模型设置好之后的固定值
            x5=[1] #一个眼睛，现在只能怪填1
            input_list.append([repeat_times,res_tag_list[i],run_time,image,x1,x2,x3,x4,x5])

    res_dic=Uget_Image_Response(input_list)
    for i in range(input_list_len):
        if res_list[i] == None:
            res_list[i]=res_dic[res_tag_list[i]]
            if not test_mod:
                np.save(res_path_list[i],res_list[i])
    return res_list

def Usave_image(axe,img_name,img_folder=""):
    """
    保存 Matplotlib 绘制的图像到本地文件系统中。

    参数：
    axe: Matplotlib 的 AxesSubplot 对象。
    img_name: 要保存的图像的名称，不包括扩展名。
    img_folder: 可选参数，指定图像保存的文件夹名称。默认为空，表示保存到默认路径。

    返回值：
    无返回值。

    注意事项：
    如果指定的文件夹不存在，则会自动创建该文件夹。
    如果没有指定文件夹，则图像将保存到默认路径。
    """
    if img_folder=="":
        img_path=img_save_path+img_name+".png"
    else:
        if not os.path.exists(img_save_path+img_folder):
            os.makedirs(img_save_path+img_folder)
        img_path=img_save_path+img_folder+"/"+img_name+".png"
    plt.savefig(img_path)
    # plt.cla()
    plt.show()

def now_time_str():#格林威治时间
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    return current_time


#读取grating的数据  file:文件路径 static_color-grid_{}_cfg.bin类型的文件
def read_grating_cfg(file):
    with open(file) as f: #可以去查查with是干嘛的，它是以防你忘记关掉文件，这种表达会比较好
        sf = np.fromfile(f,'f4',1)[0] #为啥是np.fromfile（。。。）[0],因为用np命令的读出来是array，其实我们只是要他的元素
        ori = np.fromfile(f,'f4',1)[0]
        phase = np.fromfile(f,'f4',1)[0]
        contrast = np.fromfile(f,'f4',1)[0] #一般读取到contrast就停了
        #crest = np.fromfile(f,'f4',3)  #波峰
        #valley = np.fromfile(f,'f4',3) #波谷
    return sf, ori, phase, contrast

#读取_{}.bin类型的文件，返回x_1,x_2,x_3,x_4,x_5,x_6
def read_grating_bin(file):
    with open(file) as f:
        x_1 = np.fromfile(f, 'i4', 1)  #np.array([-1]).astype('i4').tofile(f) 
        x_2 = np.fromfile(f, 'i4', 3)  #np.array([nFrame, npixel, npixel], dtype='i4').tofile(f)
        nFrame=x_2[0]
        size=x_2[1]
        x_3 = np.fromfile(f, 'f4', 3)  #mean_value.astype('f4').tofile(f) mean_value = (c1+c2)/2
        x_4 = np.fromfile(f, 'f4', 2)  #np.array([buffer_ecc, ecc], dtype='f4').tofile(f)
        x_5 = np.fromfile(f, 'u4', 1)  #np.array([neye]).astype('u4').tofile(f)
        x_6 = np.fromfile(f, 'f4', nFrame*size*size*3)  #LMS_seq.astype('f4').tofile(f)
    return x_1,x_2,x_3,x_4,x_5,x_6


#老的，不用了，但是留着，以后改了需求可能用到
# def Uget_Image_Response(image, repeat_times=1):
#     outputs=[]
#     outputs_lgn=[]
#     output_buffer = StringIO() #
#     old_stdout = sys.stdout #
#     print("=============START OURMODEL=============")
#     sys.stdout = output_buffer #
#     for i in range(repeat_times):
#         output_lgn,output = ourmodel(image)
#         outputs.append(output)
#         # output_lgn = ourmodel(input_data)[1]
#         outputs_lgn.append(output_lgn)
#     sys.stdout = old_stdout #
#     print(output_lgn[0].shape)
#     # 计算输出的平均值并返回结果
#     avg_output = sum(outputs) / len(outputs)
#     avg_output_lgn = sum(outputs_lgn) / len(outputs_lgn)
#     # np.save(data_save_path+save_name+".npy",all_results)
#     return avg_output_lgn, avg_output

# def Ubuild_IRwS_arglist(image,save_tag,repeat_times=1,run_time=1,layer="v1",size=32,frameRate=1):
#     """
#     生成并行计算所需的参数列表，使用image=这种参数语法来免除记忆的需求
#     """
#     # print([image,save_tag,int(repeat_times),run_time,layer,size,frameRate])
#     return [image,save_tag,int(repeat_times),run_time,layer,size,frameRate]

# def Ubuild_PR_arglist(C, P, SF, D, frameRate, size ,repeat_times=1,run_time=1,layer="v1"):
#     """
#     生成并行计算所需的参数列表，使用image=这种参数语法来免除记忆的需求
#     """
#     return [C, P, SF, D, frameRate, size ,int(repeat_times),run_time,layer]

print(f"UMRtools verison: {VERSION}")


