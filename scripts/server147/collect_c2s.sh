#!/bin/bash
# yvvyee: .147 c2s 결과 수집. 사용: collect_c2s.sh <model_dir: model|model_proxy> <CLASS letter> > out.txt
# 출력 줄: mode cfg name frames a kbps Y U V YUV time  (mode 는 anchor / model 로 정규화, bd_c2s.py 입력)
MD=${1:-model_proxy}; CL=${2:-D}
ssh -o ControlMaster=no -o BatchMode=yes -S /home/ywlee/.ssh/ctrl/ri147 -p 11022 yvvyee@129.254.222.147 "cd ~/SSD1/dev/c2s && for d in anchor $MD; do m=anchor; [ \$d != anchor ] && m=model; for c in RA LDB; do for f in \$d/\$c/${CL}_*_q[0-9][0-9].log; do n=\$(basename \$f .log); r=\$(grep -A1 'Total Frames |' \$f | tail -1 | tr -s ' '); t=\$(grep 'Total Time' \$f | awk '{print \$3}'); [ -n \"\$r\" ] && [ -n \"\$t\" ] && echo \"\$m \$c \$n \$r \$t\"; done; done; done"
