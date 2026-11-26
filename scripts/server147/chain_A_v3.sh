#!/bin/bash
# yvvyee: 11-26 사용자 지시로 chain_A_v2.sh 대체. 모델 A1·A2 60건을 즉시 기동하고, ParkRunning3 작업은 앵커 ParkRunning3 RA QP 22 종료 뒤에 시작한다.
# 부호화기 합 = 모델 B 28 + 앵커 1 + 모델 A 28(그중 ParkRunning3 대기 4) <= 56 물리 코어.
set -u
ROOT=/home/yvvyee/SSD1/dev/c2s; cd $ROOT
echo "CHAIN_A model A1 A2 start $(date "+%m-%d_%H:%M")" >> progress.txt
ONLY=G WAITSEQ=ParkRunning3 WAITLOG=$ROOT/anchor/RA/A2_ParkRunning3_3840x2160_50fps_10bit_420_q22.log ./c2s_run_gpu_proxy_A.sh "A1 A2" 60 28 0 "22 27 32 37 42" > run_gpu_proxy_A.log 2>&1
echo "CHAIN_A post2 A1 A2 start $(date "+%m-%d_%H:%M")" >> progress.txt
MODE=model_proxy ./c2s_post2.sh "A1 A2" 30 > post2_model_proxy_A.log 2>&1
echo "CHAIN_A DONE $(date "+%m-%d_%H:%M")" >> progress.txt
