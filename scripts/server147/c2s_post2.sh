#!/bin/bash
# yvvyee: c2s_post.sh 의 매개변수화판 (11-06). MODE 로 model(PROXY=0 실행) / model_proxy(PROXY=1 실행) 를 고른다.
#  (1) GPU 로 부호화한 모형 스트림을 CPU 복호기로 교차 복호해 부호화기 복원(GPU 복호기 재복호)과 일치하는지 확인 → crosscheck_$MODE.txt
#  (2) DecT: 앵커 스트림은 VTM-23.8 복호기, 모형 스트림은 CPU 복호기로 1스레드 복호 user 시간 → dect_$MODE.csv
#  (3) EncT: 모형을 CPU 경로 1스레드로 다시 부호화해 user 시간 → enct_cpu_$MODE/ (SKIP_ENCT=1 이면 생략)
# 사용: MODE=model_proxy [SKIP_ENCT=0] c2s_post2.sh "CLASSES" [parallel]
#   MODE=model       : TAG=modelgpu,      DEC_G=VTM-23.8_nn_gpu,       DEC_M/ENC_M=VTM-23.8_nn_avx512,    PROXY=0
#   MODE=model_proxy : TAG=modelgpuproxy, DEC_G=VTM-23.8_nn_gpu_proxy, DEC_M/ENC_M=VTM-23.8_nn_gpu_proxy, PROXY=1
set -u
CLASSES=${1:-"D"}; PAR=${2:-40}; MODE=${MODE:-model}; SKIP_ENCT=${SKIP_ENCT:-0}
ROOT=/home/yvvyee/SSD1/dev/c2s; B=/home/yvvyee/SSD1/dev
if [ "$MODE" = model_proxy ]; then TAG=modelgpuproxy; T=$B/VTM-23.8_nn_gpu_proxy; DEC_G=$T/bin/DecoderAppStatic; DEC_M=$T/bin/DecoderAppStatic; ENC_M=$T/bin/EncoderAppStatic; export VTM_NNINTERP_PROXY=1
else TAG=modelgpu; DEC_G=$B/VTM-23.8_nn_gpu/bin/DecoderAppStatic; DEC_M=$B/VTM-23.8_nn_avx512/bin/DecoderAppStatic; ENC_M=$B/VTM-23.8_nn_avx512/bin/EncoderAppStatic; export VTM_NNINTERP_PROXY=0; fi
DEC_A=$B/VTM-23.8/bin/DecoderAppStatic
EXPORT=$B/c4/export_tile_c16; YUV=$B/ctc2s; CFG=$B/VTM-23.8/cfg
POST=$ROOT/post_$MODE; ENCT=$ROOT/enct_cpu_$MODE; CROSS=$ROOT/crosscheck_$MODE.txt; DECT=$ROOT/dect_$MODE.csv
mkdir -p $POST $ENCT
one() {
  cfgname=$1; qp=$2; cls=$3; yuv=$4; pcfg=$5; fps=$6; name=${cls}_${yuv}; tag=${cfgname}_${name}_q$qp
  mb=$ROOT/$MODE/$cfgname/${name}_q$qp.bin; ab=$ROOT/anchor/$cfgname/${name}_q$qp.bin
  [ -s $mb ] && [ -s $ab ] || { echo "SKIP $tag (missing stream)"; return 0; }
  grep -q "CROSS .* $tag\$" $CROSS 2>/dev/null && grep -q "^$cfgname,$qp,$name," $DECT 2>/dev/null && { echo "skip post $tag"; } || {
  VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_GPU=1 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_yvvyee $DEC_G -b $mb -o $POST/$tag.gpu.yuv > $POST/$tag.gpudec.log 2>&1
  /usr/bin/time -f "%U" -o $POST/$tag.dect_m.txt env VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=1 $DEC_M -b $mb -o $POST/$tag.cpu.yuv > $POST/$tag.cpudec.log 2>&1
  if cmp -s $POST/$tag.gpu.yuv $POST/$tag.cpu.yuv && ! grep -q ERROR $POST/$tag.cpudec.log && [ -s $POST/$tag.cpu.yuv ]; then echo "CROSS MATCH $tag" >> $CROSS; else echo "CROSS MISMATCH $tag" >> $CROSS; fi
  rm -f $POST/$tag.gpu.yuv $POST/$tag.cpu.yuv
  /usr/bin/time -f "%U" -o $POST/$tag.dect_a.txt $DEC_A -b $ab -o /dev/null > $POST/$tag.adec.log 2>&1
  echo "$cfgname,$qp,$name,$(cat $POST/$tag.dect_a.txt),$(cat $POST/$tag.dect_m.txt)" >> $DECT
  }
  if [ "$SKIP_ENCT" != 1 ]; then
    frames=$((2*fps)); if [ $fps -ge 50 ]; then ip=64; else ip=32; fi
    if [ "$cfgname" = RA ]; then base=$CFG/encoder_randomaccess_vtm.cfg; ipopt="--IntraPeriod=$ip"; else base=$CFG/encoder_lowdelay_vtm.cfg; ipopt=""; fi
    out=$ENCT/$cfgname; mkdir -p $out; log=$out/${name}_q$qp.log
    [ -s "$log" ] && grep -q "Total Time" "$log" || VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=1 $ENC_M -c $base -c $CFG/per-sequence/$pcfg.cfg -i $YUV/$yuv.yuv -q $qp -f $frames $ipopt -b $out/${name}_q$qp.bin --SEIDecodedPictureHash=1 > $log 2>&1
    cmp -s $out/${name}_q$qp.bin $mb && echo "ENCT STREAM SAME $tag" >> $CROSS || echo "ENCT STREAM DIFF $tag" >> $CROSS
  fi
  echo "post2 done $MODE $tag $(date '+%m-%d_%H:%M')"
}
export -f one; export ROOT B MODE SKIP_ENCT DEC_A DEC_G DEC_M ENC_M EXPORT YUV CFG POST ENCT CROSS DECT
JOBS=$ROOT/jobs/jobs_post2_${MODE}_$(echo $CLASSES | tr ' ' '_').txt; : > $JOBS
for f in $ROOT/jobs/jobs_${TAG}_*.txt; do [ -f $f ] && awk -v cl=" $CLASSES " '{ if (index(cl, " "$3" ")) print $1,$2,$3,$4,$5,$6 }' $f; done | sort -u > $JOBS
echo "C2S post2 $MODE [$CLASSES] START $(date '+%m-%d_%H:%M') jobs $(wc -l < $JOBS) par $PAR skip_enct $SKIP_ENCT" >> $ROOT/progress.txt
sort -k2,2n $JOBS | xargs -P $PAR -L 1 bash -c 'one "$@"' _ >> $ROOT/progress.txt 2>&1
echo "C2S post2 $MODE [$CLASSES] DONE $(date '+%m-%d_%H:%M')" >> $ROOT/progress.txt
