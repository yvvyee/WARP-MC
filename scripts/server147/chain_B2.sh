#!/bin/bash
# yvvyee: 11-13 사용자 지시("그냥 띄워")로 chain_B.sh 를 대체. 모형 B 의 GPU 작업 32건은 체인 밖에서 post2 C 와 겹쳐 먼저 돈다(ONLY=G).
#   이 체인은 post2 C DONE 을 기다려 모형 B 의 CPU 작업 18건(ONLY=C)을 띄우고, [B G]·[B C] DONE 뒤 post2 B(par 50)를 실행한다.
# 사용: setsid nohup ./chain_B2.sh > chain_B2.log 2>&1 < /dev/null &
set -u
ROOT=/home/yvvyee/SSD1/dev/c2s; cd $ROOT
w() { until grep -q "$1" progress.txt; do sleep 600; done; }
w "^C2S post2 model_proxy \[C\] DONE"
echo "CHAIN_B2 model B cpu part start $(date "+%m-%d_%H:%M")" >> progress.txt
ONLY=C ./c2s_run_gpu_proxy.sh "B" 32 32 18 "22 27 32 37 42" > run_gpu_proxy_B_cpu.log 2>&1
w "^C2S modelgpuproxy \[B G\] DONE"; w "^C2S modelgpuproxy \[B C\] DONE"
echo "CHAIN_B2 post2 B start $(date "+%m-%d_%H:%M")" >> progress.txt
MODE=model_proxy ./c2s_post2.sh "B" 50 > post2_model_proxy_B.log 2>&1
echo "CHAIN_B2 DONE $(date "+%m-%d_%H:%M")" >> progress.txt
