#!/bin/bash
# yvvyee: 2초 CTC (각 시퀀스 앞 2×fps 프레임, QP 22/27/32/37/42, RA·LDB, 클래스 E 는 LDB 만). 앵커 VTM-23.8, 모형 VTM-23.8_nn + export_tile_c16.
# 사용: c2s_run.sh anchor|model "CLASSES" [threads] [parallel] ["QPS"]   예: c2s_run.sh model "D C E" 1 48 "22 27 32 37 42"
set -u
[ -e /home/yvvyee/SSD1/dev/c2s/HOLD ] && { echo "HOLD present, exit"; exit 0; }
MODE=$1; CLASSES=${2:-"D"}; THR=${3:-1}; PAR=${4:-48}; QPS=${5:-"22 27 32 37 42"}
ROOT=/home/yvvyee/SSD1/dev/c2s; YUV=/home/yvvyee/SSD1/dev/ctc2s; CFG=/home/yvvyee/SSD1/dev/VTM-23.8/cfg
ENC_A=/home/yvvyee/SSD1/dev/VTM-23.8/bin/EncoderAppStatic
ENC_M=/home/yvvyee/SSD1/dev/VTM-23.8_nn/bin/EncoderAppStatic; DEC_M=/home/yvvyee/SSD1/dev/VTM-23.8_nn/bin/DecoderAppStatic
EXPORT=/home/yvvyee/SSD1/dev/c4/export_tile_c16
# class:yuvname:per-seq cfg:fps  (frames = 2*fps, RA IntraPeriod 32 for 24/25/30 fps, 64 for 50/60 fps)
SEQS="D:BasketballPass_416x240_50:BasketballPass:50 D:BQSquare_416x240_60:BQSquare:60 D:BlowingBubbles_416x240_50:BlowingBubbles:50 D:RaceHorses_416x240_30:RaceHorses:30
C:BasketballDrill_832x480_50:BasketballDrill:50 C:BQMall_832x480_60:BQMall:60 C:PartyScene_832x480_50:PartyScene:50 C:RaceHorses_832x480_30:RaceHorsesC:30
E:FourPeople_1280x720_60:FourPeople:60 E:Johnny_1280x720_60:Johnny:60 E:KristenAndSara_1280x720_60:KristenAndSara:60
B:Cactus_1920x1080_50:Cactus:50 B:BasketballDrive_1920x1080_50:BasketballDrive:50 B:BQTerrace_1920x1080_60:BQTerrace:60 B:MarketPlace_1920x1080_60fps_10bit_420:MarketPlace:60 B:RitualDance_1920x1080_60fps_10bit_420:RitualDance:60
A1:Tango2_3840x2160_60fps_10bit_420:Tango2:60 A1:FoodMarket4_3840x2160_60fps_10bit_420:FoodMarket4:60 A1:Campfire_3840x2160_30fps_10bit_420_bt709_videoRange:Campfire:30
A2:CatRobot_3840x2160_60fps_10bit_420_jvet:CatRobot:60 A2:DaylightRoad2_3840x2160_60fps_10bit_420:DaylightRoad2:60 A2:ParkRunning3_3840x2160_50fps_10bit_420:ParkRunning3:50"
mkdir -p $ROOT/jobs; JOBS=$ROOT/jobs/jobs_${MODE}_$(echo $CLASSES | tr ' ' '_').txt; : > $JOBS
for s in $SEQS; do
  IFS=: read cls yuv pcfg fps <<< "$s"
  echo " $CLASSES " | grep -q " $cls " || continue
  [ -f $CFG/per-sequence/$pcfg.cfg ] || { echo "MISSING cfg $pcfg"; continue; }
  for cfgname in RA LDB; do
    [ "$cls" = E ] && [ "$cfgname" = RA ] && continue
    for qp in $QPS; do echo "$cfgname $qp $cls $yuv $pcfg $fps" >> $JOBS; done
  done
done
one() {
  cfgname=$1; qp=$2; cls=$3; yuv=$4; pcfg=$5; fps=$6
  frames=$((2*fps)); if [ $fps -ge 50 ]; then ip=64; else ip=32; fi
  out=$ROOT/$MODE/$cfgname; mkdir -p $out; name=${cls}_${yuv}
  log=$out/${name}_q$qp.log; bin=$out/${name}_q$qp.bin
  [ -s "$log" ] && grep -q "Total Time" "$log" && { echo "skip $MODE $cfgname $name q$qp"; return 0; }
  if [ "$cfgname" = RA ]; then base=$CFG/encoder_randomaccess_vtm.cfg; ipopt="--IntraPeriod=$ip"; else base=$CFG/encoder_lowdelay_vtm.cfg; ipopt=""; fi
  common="-c $base -c $CFG/per-sequence/$pcfg.cfg -i $YUV/$yuv.yuv -q $qp -f $frames $ipopt -b $bin --SEIDecodedPictureHash=1"
  if [ "$MODE" = anchor ]; then
    $ENC_A $common > $log 2>&1 || echo "FAIL anchor $cfgname $name q$qp"
  else
    rec=$out/${name}_q$qp.rec.yuv
    VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=$THR $ENC_M $common -o $rec > $log 2>&1 || { echo "FAIL model $cfgname $name q$qp"; return 1; }
    VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=$THR $DEC_M -b $bin -o $rec.dec -d 10 > $out/${name}_q$qp.dec.log 2>&1
    if cmp -s $rec $rec.dec && ! grep -q "ERROR" $out/${name}_q$qp.dec.log; then echo "MATCH $cfgname $name q$qp" >> $ROOT/$MODE/consistency.txt; rm -f $rec $rec.dec; else echo "MISMATCH $cfgname $name q$qp" >> $ROOT/$MODE/consistency.txt; fi
  fi
  echo "done $MODE $cfgname $name q$qp $(date '+%m-%d_%H:%M')"
}
export -f one; export MODE THR ROOT YUV CFG ENC_A ENC_M DEC_M EXPORT
echo "C2S $MODE [$CLASSES] START $(date '+%m-%d_%H:%M') jobs $(wc -l < $JOBS) threads $THR par $PAR" >> $ROOT/progress.txt
# 긴 작업(낮은 QP, 큰 클래스) 먼저
sort -k3,3 -k2,2n $JOBS | xargs -P $PAR -L 1 bash -c 'one "$@"' _ >> $ROOT/progress.txt 2>&1
echo "C2S $MODE [$CLASSES] DONE $(date '+%m-%d_%H:%M')" >> $ROOT/progress.txt
