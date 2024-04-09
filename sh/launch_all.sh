#!/usr/bin/env bash
source ~/anaconda3/bin/activate torch

# 在当前目录执行python指令，并保存PID
pids=()
python ../src/client.py --client=0 & pids+=($!)
python ../src/client.py --client=1 & pids+=($!)
python ../src/client.py --client=2 & pids+=($!)
python ../src/client.py --client=3 & pids+=($!)
python ../src/client.py --client=4 & pids+=($!)

# 在脚本退出时终止所有的client.py后台进程
trap 'kill ${pids[@]}' EXIT

# 等待所有后台进程完成
wait

