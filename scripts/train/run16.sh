#!/bin/bash
# yvvyee: L7 16채널 타일 모형 재학습 (2026-11-04 사용자 승인, stage4 13.8절). 시드별 GPU 1 장, 3 시드 병렬. 나머지 설정은 c32 와 동일 (nl_tile.py 기본값).
cd /home/ywlee/dev/oracle_runs/bli
P=~/miniforge3/envs/oracle/bin/python
s=$1; g=$2
echo "start TILE16 s$s gpu$g $(date '+%m-%d_%H:%M')" >> queue16.txt
CUDA_VISIBLE_DEVICES=$g CH=16 SEED=$s OUTDIR=eval5 $P nl_tile.py > eval5/log_TILE16_s$s.txt 2>&1 && echo "done 0 TILE16 s$s $(date '+%m-%d_%H:%M')" >> queue16.txt || echo "done $? TILE16 s$s $(date '+%m-%d_%H:%M')" >> queue16.txt
