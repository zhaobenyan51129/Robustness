import os


patchV1_dir='/home/zhaobenyan/repos/patchV1/'
src_dir='/home/zhaobenyan/repos/patchV1/src/'
setup_dir='/home/zhaobenyan/dataset/setup/'
patchfast='/home/zhaobenyan/dataset/patchfast/'

#以下操作都只需要运行一次，每一步生成文件要存在哪个文件夹就cd哪个文件夹
#进入 $HOME/repos/patchV1/src 分别执行 compile-nonhpc、gompile-nonhpc 和 rompile-nonhpc 脚本
def compile():
    os.system(f'cd {src_dir}/ && ./compile-nonhpc')
    os.system(f'cd {src_dir}/ && ./gompile-nonhpc')
    os.system(f'cd {src_dir}/ && ./rompile-nonhpc')

#运行pinwheel_disk_tmp.py 生成resource文件存在/home/zhaobenyan/dataset/resource
def resource():
    os.system(f'cd {patchV1_dir}/ && python pinwheel_disk.py')

# 运行retino和genCon,resourceFolder和inputFolder需要修改,生成文件储存在setup
def retine():
    os.system(f'cd {setup_dir} && retino -c {src_dir}test_lgn.cfg')
    os.system(f'cd {setup_dir} && genCon -c {src_dir}test_v1.cfg')


#运行patchfast，生成文件储存在patchfast
def patch():
    os.system(f'cd {patchfast} && patch_fast -c {src_dir}minimal.cfg') 

#运行minimalTC
#可以在看log文件，观察是哪里报错了
def minimalTC():
    os.system(f'cd {src_dir} && ./minimalTC')

def main():
    #compile()
    #resource()
    #retine()
    #patch()
    minimalTC()

main()

print("complete!!")