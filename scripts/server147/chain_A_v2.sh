#!/bin/bash
# yvvyee: 11-25 chain_A.sh 대체. chain_A.sh 는 앵커 A2 나머지 QP 를 앵커 A1 DONE 뒤에 띄워, 앵커 A1 이 줄어드는 동안 코어가 비었다(04:46 에 앵커 14 + 모델 29 = 43/56).
#   앵커 A2 나머지 QP(24건)를 지금 par 12 로 띄운다(앵커 14 + 12 = 26 이하 유지). 앵커 A2 QP 22(6건)는 chain_A.sh 가 띄운 실행이 그대로 돈다.
#   앵커 A1 DONE, 앵커 A2 DONE 2회(QP 22 실행과 나머지 실행)를 확인한 뒤 모델 A1·A2 60건(GPU 경로, PG 26)과 post2 A1·A2(par 30)를 실행한다.
# 사용: setsid nohup ./chain_A_v2.sh > chain_A_v2.log 2>&1 < /dev/null &
set -u
ROOT=/home/yvvyee/SSD1/dev/c2s; cd $ROOT
echo "CHAIN_A anchor A2 rest start $(date "+%m-%d_%H:%M")" >> progress.txt
./c2s_run.sh anchor "A2" 1 12 "27 32 37 42" > launch_A2_anchor_rest.log 2>&1
until grep -q "^C2S anchor \[A1\] DONE" progress.txt && [ "$(grep -c "^C2S anchor \[A2\] DONE" progress.txt)" -ge 2 ]; do sleep 600; done
echo "CHAIN_A model A1 A2 start $(date "+%m-%d_%H:%M")" >> progress.txt
ONLY=G ./c2s_run_gpu_proxy_A.sh "A1 A2" 60 26 0 "22 27 32 37 42" > run_gpu_proxy_A.log 2>&1
echo "CHAIN_A post2 A1 A2 start $(date "+%m-%d_%H:%M")" >> progress.txt
MODE=model_proxy ./c2s_post2.sh "A1 A2" 30 > post2_model_proxy_A.log 2>&1
echo "CHAIN_A DONE $(date "+%m-%d_%H:%M")" >> progress.txt
