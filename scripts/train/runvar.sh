#!/bin/bash
# yvvyee: 타일 모형 축소 재학습 (2026-11-04 15:05 사용자 승인, 2단계 고속화 4항). 사용: runvar.sh CH DEPTH SEED GPU  (나머지 설정은 c32 와 동일)
cd /home/ywlee/dev/oracle_runs/bli
P=~/miniforge3/envs/oracle/bin/python
ch=$1; d=$2; s=$3; g=$4; tag=TILE_c${ch}_d${d}_s$s
echo "start $tag gpu$g $(date '+%m-%d_%H:%M')" >> queuevar.txt
CUDA_VISIBLE_DEVICES=$g CH=$ch DEPTH=$d SEED=$s OUTDIR=eval5 $P nl_tile.py > eval5/log_$tag.txt 2>&1 && echo "done 0 $tag $(date '+%m-%d_%H:%M')" >> queuevar.txt || echo "done $? $tag $(date '+%m-%d_%H:%M')" >> queuevar.txt
