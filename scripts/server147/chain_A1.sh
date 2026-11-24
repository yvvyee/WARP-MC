#!/bin/bash
# yvvyee: 11-24 사용자 지시("가능하다면 클래스 A1, A2 도 시작해")로 클래스 A1 체인. A2 는 2초 클립이 서버에 없어 제외.
#   앵커 A1(체인 밖, par 26: 모델 B 30 + 26 = 56 물리 코어) DONE 을 기다려 모델 A1 30건을 전부 GPU 경로(ONLY=G, PG 26)로 띄우고, 끝나면 post2 A1(par 30)을 실행한다.
# 사용: setsid nohup ./chain_A1.sh > chain_A1.log 2>&1 < /dev/null &
set -u
ROOT=/home/yvvyee/SSD1/dev/c2s; cd $ROOT
w() { until grep -q "$1" progress.txt; do sleep 600; done; }
w "^C2S anchor \[A1\] DONE"
echo "CHAIN_A1 model A1 start $(date "+%m-%d_%H:%M")" >> progress.txt
ONLY=G ./c2s_run_gpu_proxy_A.sh "A1" 30 26 0 "22 27 32 37 42" > run_gpu_proxy_A1.log 2>&1
echo "CHAIN_A1 post2 A1 start $(date "+%m-%d_%H:%M")" >> progress.txt
MODE=model_proxy ./c2s_post2.sh "A1" 30 > post2_model_proxy_A1.log 2>&1
echo "CHAIN_A1 DONE $(date "+%m-%d_%H:%M")" >> progress.txt
