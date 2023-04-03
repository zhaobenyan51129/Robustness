#!/bin/bash

# 指定要删除文件的目录
dir_path="/home/zhaobenyan/model/data_save/log"

# 执行删除操作
find "$dir_path" -type f -mtime +15 -exec rm {} \;

#若要取消：crontab -e 命令打开crontab编辑器，然后删除或注释掉对应的定时任务行