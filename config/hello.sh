#! /bin/bash

# # $#获取参数总个数  -ne 不等于
# :<<!  
# [ $# -ne 2 ] && {
#       echo "must be two args"
#       exit 119  #终止程序运行，且返回119状态码，提供给当前shell的$?变量
# }
# echo ok
# echo "当前脚本的id是: $$"  #$$返回当前脚本的id
# !

# :<<!
# #函数
# print_usage(){
#       printf "Please enter an integer!!!\n"
#       # 给脚本执行结果赋予一个状态码
#       exit 3
# }

# #read -p "提示信息" 接受用户输入的变量
# read -p "Please input your number" firstnum
# #-n 判断字符串是否为空
# #sed 将firstnum中的数字替换为空，剩下非数字的内容
# if [ -n "`echo $firstnum|sed 's/[0-9]//g'`" ]
#       then  #非空
#             print_usage
# fi
# !


# :<<!
# #判断文件名后缀是否合法
# if expr "$1" ":" ".*\.jpg" &> /dev/null
#       then
#             echo "这是一个jpg文件"
# else
#       echo "这不是jpg文件"
# fi
# !

# # 删除指定文件夹下（Shell文件夹）所有test.txt
# #find 文件夹名 -name test.txt |xargs rm -rf

#替换指定目录下的文件名 sed 's/pattern/replace/' 末尾添加g用于替换所有匹配项，而不仅仅替换第一个匹配项。
cd /home/zhaobenyan/dataset/output/driftgrating_32x32/heatmap
for file in `ls | grep .png`
do
      newfile=`echo $file | sed 's/:/_/g'`
      mv $file $newfile
done


