#!/bin/bash
# yvvyee: 11-09 클래스 B 체인. 앵커 B DONE 과 모형 C DONE 을 기다린 뒤 post2 C(par 40) → 모형 B GPU 실행(32 GPU/18 CPU, PROXY=1) → post2 B(par 50) 를 순서대로 실행한다.
#   EncT 측정(post2 의 CPU 1스레드 재부호화)이 다른 부호화 작업과 겹치지 않도록 직렬로 둔다. 앵커 B 는 체인 밖에서 par 16 으로 먼저 돈다(C 모형 40 + 16 = 56 물리 코어).
# 사용: setsid nohup ./chain_B.sh > chain_B.log 2>&1 < /dev/null &
set -u
ROOT=/home/yvvyee/SSD1/dev/c2s; cd $ROOT
w() { until grep -q "$1" progress.txt; do sleep 600; done; }
w "^C2S anchor \[B\] DONE"; w "^C2S modelgpuproxy \[C\] DONE"
echo "CHAIN_B post2 C start $(date "+%m-%d_%H:%M")" >> progress.txt
MODE=model_proxy ./c2s_post2.sh "C" 40 > post2_model_proxy_C.log 2>&1
echo "CHAIN_B model B start $(date "+%m-%d_%H:%M")" >> progress.txt
./c2s_run_gpu_proxy.sh "B" 32 32 18 "22 27 32 37 42" > run_gpu_proxy_B.log 2>&1
echo "CHAIN_B post2 B start $(date "+%m-%d_%H:%M")" >> progress.txt
MODE=model_proxy ./c2s_post2.sh "B" 50 > post2_model_proxy_B.log 2>&1
echo "CHAIN_B DONE $(date "+%m-%d_%H:%M")" >> progress.txt
