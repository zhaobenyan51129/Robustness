#!/bin/bash

#source执行脚本 (不会开启子shell)
source /home/zhaobenyan/miniconda3/etc/profile.d/conda.sh
# eval 执行多个脚本  用shell脚本激活conda虚拟环境
eval "$(conda shell.bash hook)"
conda activate neuro

repos=$HOME/repos
dataset=$HOME/dataset
bin=$HOME/bin
resource=$dataset/resource
setup=$dataset/setup
patchfast=$dataset/patchfast
grating=$dataset/grating
output=$dataset/output
src=$HOME/repos/patchV1/src
txt=$dataset/Documentation.txt

echo Start creating directory ...
cd $HOME   #进入根目录/home/zhaobenyan
#创建一个文件夹dataset储存戴老师代码所有的数据
for dir in $bin $dataset $resource $setup $patchfast $grating $output 
do
      if [ -d $dir ];then    #-d检测文件是否是目录且是否存在，如果是目录且存在返回true
            echo $dir exists!
      else
            mkdir -p $dir	 #如果不存在,创建（-p按层级创建）
            echo $dir Created!
      fi
done

#创建说明文档Documentation.txt对每个文件的内容进行说明
echo "Begin to write our own Documentation.txt"
 
cat>$txt<<EOF
bin:存放compile、gompile和rompile脚本生成的patch_fast、genCon和retino文件
resource:存放pinwheel_disk_tmp.py生产的数据
setup:retino和genCon产生的数据
patchfast:patch_fast(minimal.cfg、minimalTC) 产生的数据
grating:存放/to_configure/generate_grating.py生成的grating
output:存放ouemodel实验生成的数据
EOF

#创建repos目录存放戴老师代码
if [ -d "$repos" ];then    #-d检测文件是否是目录且是否存在，如果是目录且存在返回true
	echo $repos exists!
else
      mkdir -p $repos	  #如果不存在,创建（-p按层级创建）
      echo $repos Created!
fi

# 后台运行脚本clone代码
cd $repos
if [ ! -d "patchV1" ];then
      git init
      echo "start clone ..."
      git clone https://github.com/g13/patchV1.git &
      wait
      echo "Clone complete!"
      sleep 3;cd $repos/patchV1;pwd  #这里要cd之后再切换分支，否则会报错
      git pull origin minimal;git checkout -b minimal;git branch
else
      echo "The code has been cloned!"
fi