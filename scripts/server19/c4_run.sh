#!/bin/bash
# yvvyee: C4 축소 CTC (클래스 D·C, 32프레임, QP 22/27/32/37, RA·LDB) — 앵커(VTM-23.8)와 모형(VTM-23.8_nn + export_tile_s0) 부호화.
# 사용: c4_run.sh anchor|model [threads]   (앵커를 먼저 전부 돌린 뒤 모형을 돌린다. 작업 목록은 긴 것부터.)
set -u
[ -e /home/ywlee/dev/c4/HOLD ] && { echo "HOLD present, exit"; exit 0; }
MODE=$1; THR=${2:-1}
ROOT=/home/ywlee/dev/c4; YUV=/home/ywlee/dataset/CTC/orig/yuv; CFG=/home/ywlee/dev/VTM-23.8/cfg
ENC_A=/home/ywlee/dev/VTM-23.8/bin/EncoderAppStatic
ENC_M=/home/ywlee/dev/VTM-23.8_nn/bin/EncoderAppStatic; DEC_M=/home/ywlee/dev/VTM-23.8_nn/bin/DecoderAppStatic
EXPORT=$ROOT/export_tile_s0
FRAMES=32
# 시퀀스: 이름 cfg이름 인트라주기(VTM CTC: 24/25/30 fps 32, 50/60 fps 64; GOP 32 의 배수여야 한다)
SEQS="C_BasketballDrill_832x480_50:BasketballDrill:64 C_BQMall_832x480_60:BQMall:64 C_PartyScene_832x480_50:PartyScene:64 C_RaceHorses_832x480_30:RaceHorsesC:32 D_BQSquare_416x240_60:BQSquare:64 D_BasketballPass_416x240_50:BasketballPass:64 D_BlowingBubbles_416x240_50:BlowingBubbles:64 D_RaceHorses_416x240_30:RaceHorses:32"
mkdir -p $ROOT/jobs
JOBS=$ROOT/jobs/jobs_$MODE.txt; : > $JOBS
for cfgname in RA LDB; do
  for qp in 22 27 32 37; do
    for s in $SEQS; do
      name=${s%%:*}; rest=${s#*:}; pcfg=${rest%%:*}; ip=${rest#*:}
      echo "$cfgname $qp $name $pcfg $ip" >> $JOBS
    done
  done
done
one() {
  cfgname=$1; qp=$2; name=$3; pcfg=$4; ip=$5
  out=$ROOT/$MODE/$cfgname; mkdir -p $out
  log=$out/${name}_q$qp.log; bin=$out/${name}_q$qp.bin
  [ -s "$log" ] && grep -q "Total Time" "$log" && { echo "skip $MODE $cfgname $name q$qp"; return 0; }
  if [ "$cfgname" = RA ]; then base=$CFG/encoder_randomaccess_vtm.cfg; ipopt="--IntraPeriod=$ip"; else base=$CFG/encoder_lowdelay_vtm.cfg; ipopt=""; fi
  common="-c $base -c $CFG/per-sequence/$pcfg.cfg -i $YUV/${name#?_}.yuv -q $qp -f $FRAMES $ipopt -b $bin --SEIDecodedPictureHash=1"
  if [ "$MODE" = anchor ]; then
    $ENC_A $common > $log 2>&1 || echo "FAIL anchor $cfgname $name q$qp"
  else
    rec=$out/${name}_q$qp.rec.yuv
    VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=$THR $ENC_M $common -o $rec > $log 2>&1 || { echo "FAIL model $cfgname $name q$qp"; return 1; }
    # 복호기 일치 검사: 모형 복호기 복원 = 부호화기 복원, 화면 해시 오류 0
    VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=$THR $DEC_M -b $bin -o $rec.dec -d 10 > $out/${name}_q$qp.dec.log 2>&1
    if cmp -s $rec $rec.dec && ! grep -q "ERROR" $out/${name}_q$qp.dec.log; then echo "MATCH $cfgname $name q$qp" >> $ROOT/$MODE/consistency.txt; rm -f $rec $rec.dec; else echo "MISMATCH $cfgname $name q$qp" >> $ROOT/$MODE/consistency.txt; fi
  fi
  echo "done $MODE $cfgname $name q$qp $(date '+%m-%d_%H:%M')"
}
export -f one; export MODE THR ROOT YUV CFG ENC_A ENC_M DEC_M EXPORT FRAMES
echo "C4 $MODE START $(date '+%m-%d_%H:%M') jobs $(wc -l < $JOBS) threads $THR" >> $ROOT/progress.txt
# 클래스 D 먼저(결과를 빨리 얻기 위해), 같은 클래스 안에서는 낮은 QP(긴 작업) 먼저
sort -k3,3r -k2,2n $JOBS | awk '{print}' | xargs -P 20 -L 1 bash -c 'one "$@"' _ >> $ROOT/progress.txt 2>&1
echo "C4 $MODE DONE $(date '+%m-%d_%H:%M')" >> $ROOT/progress.txt
