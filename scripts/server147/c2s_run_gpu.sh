#!/bin/bash
# yvvyee: c2s_run.sh 의 model 모드 GPU 혼합판. 작업 생성·skip·일치 검사 논리는 c2s_run.sh 와 같다.
#   정렬(긴 작업 먼저) 후 앞 NG 건은 GPU 경로(VTM-23.8_nn_gpu, VTM_NNINTERP_GPU=1, 장치 = 순번 mod 2), 나머지는 CPU 경로(VTM-23.8_nn_avx512).
#   GPU 작업은 MPS 환경(CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_yvvyee)으로 실행. 출력은 model/ (앵커 비교용), 진행 기록 표지는 modelgpu.
# 사용: c2s_run_gpu.sh "CLASSES" [NG=32] [PG=32] [PC=8] ["QPS"] [--dry]
set -u
[ -e /home/yvvyee/SSD1/dev/c2s/HOLD ] && { echo "HOLD present, exit"; exit 0; }
CLASSES=${1:-"D"}; NG=${2:-32}; PG=${3:-32}; PC=${4:-8}; QPS=${5:-"22 27 32 37 42"}; DRY=${6:-}
MODE=model; TAG=modelgpu; THR=1
ROOT=/home/yvvyee/SSD1/dev/c2s; YUV=/home/yvvyee/SSD1/dev/ctc2s; CFG=/home/yvvyee/SSD1/dev/VTM-23.8/cfg
ENC_G=/home/yvvyee/SSD1/dev/VTM-23.8_nn_gpu/bin/EncoderAppStatic; DEC_G=/home/yvvyee/SSD1/dev/VTM-23.8_nn_gpu/bin/DecoderAppStatic
ENC_C=/home/yvvyee/SSD1/dev/VTM-23.8_nn_avx512/bin/EncoderAppStatic; DEC_C=/home/yvvyee/SSD1/dev/VTM-23.8_nn_avx512/bin/DecoderAppStatic
EXPORT=/home/yvvyee/SSD1/dev/c4/export_tile_c16
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_yvvyee CUDA_MPS_LOG_DIRECTORY=/tmp/mps_yvvyee_log
SEQS="D:BasketballPass_416x240_50:BasketballPass:50 D:BQSquare_416x240_60:BQSquare:60 D:BlowingBubbles_416x240_50:BlowingBubbles:50 D:RaceHorses_416x240_30:RaceHorses:30
C:BasketballDrill_832x480_50:BasketballDrill:50 C:BQMall_832x480_60:BQMall:60 C:PartyScene_832x480_50:PartyScene:50 C:RaceHorses_832x480_30:RaceHorsesC:30
E:FourPeople_1280x720_60:FourPeople:60 E:Johnny_1280x720_60:Johnny:60 E:KristenAndSara_1280x720_60:KristenAndSara:60"
mkdir -p $ROOT/jobs; JOBS=$ROOT/jobs/jobs_${TAG}_$(echo $CLASSES | tr ' ' '_').txt; : > $JOBS
for s in $SEQS; do
  IFS=: read cls yuv pcfg fps <<< "$s"
  echo " $CLASSES " | grep -q " $cls " || continue
  [ -f $CFG/per-sequence/$pcfg.cfg ] || { echo "MISSING cfg $pcfg"; continue; }
  for cfgname in RA LDB; do
    [ "$cls" = E ] && [ "$cfgname" = RA ] && continue
    for qp in $QPS; do echo "$cfgname $qp $cls $yuv $pcfg $fps" >> $JOBS; done
  done
done
# 긴 작업(낮은 QP, 큰 클래스) 먼저, 순번을 붙여 앞 NG 건 GPU / 나머지 CPU
SORTED=$JOBS.sorted; sort -k3,3 -k2,2n $JOBS | awk -v ng=$NG '{print (NR<=ng ? "G " (NR-1)%2 : "C -"), $0}' > $SORTED
one() {
  path=$1; dev=$2; cfgname=$3; qp=$4; cls=$5; yuv=$6; pcfg=$7; fps=$8
  frames=$((2*fps)); if [ $fps -ge 50 ]; then ip=64; else ip=32; fi
  out=$ROOT/$MODE/$cfgname; mkdir -p $out; name=${cls}_${yuv}
  log=$out/${name}_q$qp.log; bin=$out/${name}_q$qp.bin; rec=$out/${name}_q$qp.rec.yuv
  [ -s "$log" ] && grep -q "Total Time" "$log" && { echo "skip $TAG $cfgname $name q$qp"; return 0; }
  if [ "$cfgname" = RA ]; then base=$CFG/encoder_randomaccess_vtm.cfg; ipopt="--IntraPeriod=$ip"; else base=$CFG/encoder_lowdelay_vtm.cfg; ipopt=""; fi
  common="-c $base -c $CFG/per-sequence/$pcfg.cfg -i $YUV/$yuv.yuv -q $qp -f $frames $ipopt -b $bin --SEIDecodedPictureHash=1"
  if [ "$path" = G ]; then
    VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=$THR VTM_NNINTERP_GPU=1 VTM_NNINTERP_GPU_DEV=$dev $ENC_G $common -o $rec > $log 2>&1 || { echo "FAIL $TAG gpu $cfgname $name q$qp"; return 1; }
    VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=$THR VTM_NNINTERP_GPU=1 VTM_NNINTERP_GPU_DEV=$dev $DEC_G -b $bin -o $rec.dec -d 10 > $out/${name}_q$qp.dec.log 2>&1
  else
    VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=$THR $ENC_C $common -o $rec > $log 2>&1 || { echo "FAIL $TAG cpu $cfgname $name q$qp"; return 1; }
    VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=$THR $DEC_C -b $bin -o $rec.dec -d 10 > $out/${name}_q$qp.dec.log 2>&1
  fi
  if cmp -s $rec $rec.dec && ! grep -q "ERROR" $out/${name}_q$qp.dec.log; then echo "MATCH $cfgname $name q$qp ($path)" >> $ROOT/$MODE/consistency.txt; rm -f $rec $rec.dec; else echo "MISMATCH $cfgname $name q$qp ($path)" >> $ROOT/$MODE/consistency.txt; fi
  echo "done $TAG $cfgname $name q$qp $path$dev $(date '+%m-%d_%H:%M')"
}
export -f one; export MODE TAG THR ROOT YUV CFG ENC_G DEC_G ENC_C DEC_C EXPORT CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
if [ "$DRY" = "--dry" ]; then echo "GPU jobs: $(grep -c '^G' $SORTED) (dev0 $(grep -c '^G 0' $SORTED), dev1 $(grep -c '^G 1' $SORTED)), CPU jobs: $(grep -c '^C' $SORTED)"; head -3 $SORTED; tail -9 $SORTED | head -2; exit 0; fi
echo "C2S $TAG [$CLASSES] START $(date '+%m-%d_%H:%M') jobs $(wc -l < $JOBS) gpu $(grep -c '^G' $SORTED)/P$PG cpu $(grep -c '^C' $SORTED)/P$PC" >> $ROOT/progress.txt
grep '^G' $SORTED | xargs -P $PG -L 1 bash -c 'one "$@"' _ >> $ROOT/progress.txt 2>&1 &
grep '^C' $SORTED | xargs -P $PC -L 1 bash -c 'one "$@"' _ >> $ROOT/progress.txt 2>&1 &
wait
echo "C2S $TAG [$CLASSES] DONE $(date '+%m-%d_%H:%M')" >> $ROOT/progress.txt
