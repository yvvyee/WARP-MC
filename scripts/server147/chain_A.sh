#!/bin/bash
# yvvyee: 11-24 사용자 지시("A2 영상 가져와")로 chain_A1.sh 를 대체하는 클래스 A1·A2 체인.
#   앵커 시간(EncT 분모)이 물리 코어 56개를 넘는 조건에서 측정되지 않도록 앵커 프로세스 합을 26 이하로 둔다(모델 B 30 + 26 = 56).
#   앵커 A1(체인 밖, par 26)의 빈 자리 6개에 앵커 A2 QP 22(6건)를 먼저 넣고, 앵커 A1 DONE 뒤 앵커 A2 나머지 QP(24건, par 20)를 돌린다.
#   앵커 A2 가 모두 끝나면 모델 A1·A2 60건을 전부 GPU 경로(ONLY=G, PG 26)로 띄우고, 끝나면 post2 A1·A2(par 30)를 실행한다.
# 사용: setsid nohup ./chain_A.sh > chain_A.log 2>&1 < /dev/null &
set -u
ROOT=/home/yvvyee/SSD1/dev/c2s; cd $ROOT
w() { until grep -q "$1" progress.txt; do sleep 600; done; }
echo "CHAIN_A anchor A2 q22 start $(date "+%m-%d_%H:%M")" >> progress.txt
./c2s_run.sh anchor "A2" 1 6 "22" > launch_A2_anchor_q22.log 2>&1 &
w "^C2S anchor \[A1\] DONE"
echo "CHAIN_A anchor A2 rest start $(date "+%m-%d_%H:%M")" >> progress.txt
./c2s_run.sh anchor "A2" 1 20 "27 32 37 42" > launch_A2_anchor_rest.log 2>&1
wait
echo "CHAIN_A model A1 A2 start $(date "+%m-%d_%H:%M")" >> progress.txt
ONLY=G ./c2s_run_gpu_proxy_A.sh "A1 A2" 60 26 0 "22 27 32 37 42" > run_gpu_proxy_A.log 2>&1
echo "CHAIN_A post2 A1 A2 start $(date "+%m-%d_%H:%M")" >> progress.txt
MODE=model_proxy ./c2s_post2.sh "A1 A2" 30 > post2_model_proxy_A.log 2>&1
echo "CHAIN_A DONE $(date "+%m-%d_%H:%M")" >> progress.txt
