#!/bin/bash
# yvvyee: 2초 프로토콜 사후 처리 (사용자 지시 11-04 17:45: EncT·DecT 모두 보고, 조건 명시).
#  (1) GPU 로 부호화한 모형 스트림을 CPU 복호기(AVX-512 빌드)로 교차 복호해 부호화기 복원과 일치하는지 확인 → crosscheck.txt
#  (2) DecT: 앵커 스트림은 VTM-23.8 복호기, 모형 스트림은 CPU 복호기로 1스레드 복호 user 시간 → dect.csv
#  (3) EncT: 모형을 CPU 경로(AVX-512 빌드) 1스레드로 다시 부호화해 user 시간 → enct_cpu/ (BD-rate 용 스트림은 GPU 실행 것을 쓰고, 시간만 여기서)
# 사용: c2s_post.sh "CLASSES" [parallel]
set -u
CLASSES=${1:-"D"}; PAR=${2:-40}
ROOT=/home/yvvyee/SSD1/dev/c2s; B=/home/yvvyee/SSD1/dev
DEC_A=$B/VTM-23.8/bin/DecoderAppStatic; DEC_M=$B/VTM-23.8_nn_avx512/bin/DecoderAppStatic; ENC_M=$B/VTM-23.8_nn_avx512/bin/EncoderAppStatic
EXPORT=$B/c4/export_tile_c16; YUV=$B/ctc2s; CFG=$B/VTM-23.8/cfg
mkdir -p $ROOT/post $ROOT/enct_cpu
one() {
  cfgname=$1; qp=$2; cls=$3; yuv=$4; pcfg=$5; fps=$6; name=${cls}_${yuv}; tag=${cfgname}_${name}_q$qp
  mb=$ROOT/model/$cfgname/${name}_q$qp.bin; ab=$ROOT/anchor/$cfgname/${name}_q$qp.bin
  [ -s $mb ] && [ -s $ab ] || { echo "SKIP $tag (missing stream)"; return 0; }
  # (1) cross decode with CPU decoder; encoder recon was deleted after MATCH, so re-derive it by decoding with the GPU decoder once more
  VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_GPU=1 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_yvvyee $B/VTM-23.8_nn_gpu/bin/DecoderAppStatic -b $mb -o $ROOT/post/$tag.gpu.yuv > $ROOT/post/$tag.gpudec.log 2>&1
  /usr/bin/time -f "%U" -o $ROOT/post/$tag.dect_m.txt env VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=1 $DEC_M -b $mb -o $ROOT/post/$tag.cpu.yuv > $ROOT/post/$tag.cpudec.log 2>&1
  if cmp -s $ROOT/post/$tag.gpu.yuv $ROOT/post/$tag.cpu.yuv && ! grep -q ERROR $ROOT/post/$tag.cpudec.log; then echo "CROSS MATCH $tag" >> $ROOT/crosscheck.txt; else echo "CROSS MISMATCH $tag" >> $ROOT/crosscheck.txt; fi
  rm -f $ROOT/post/$tag.gpu.yuv $ROOT/post/$tag.cpu.yuv
  # (2) anchor DecT
  /usr/bin/time -f "%U" -o $ROOT/post/$tag.dect_a.txt $DEC_A -b $ab -o /dev/null > $ROOT/post/$tag.adec.log 2>&1
  echo "$cfgname,$qp,$name,$(cat $ROOT/post/$tag.dect_a.txt),$(cat $ROOT/post/$tag.dect_m.txt)" >> $ROOT/dect.csv
  # (3) CPU-path EncT (single thread), stream must equal the GPU stream
  frames=$((2*fps)); if [ $fps -ge 50 ]; then ip=64; else ip=32; fi
  if [ "$cfgname" = RA ]; then base=$CFG/encoder_randomaccess_vtm.cfg; ipopt="--IntraPeriod=$ip"; else base=$CFG/encoder_lowdelay_vtm.cfg; ipopt=""; fi
  out=$ROOT/enct_cpu/$cfgname; mkdir -p $out; log=$out/${name}_q$qp.log
  [ -s "$log" ] && grep -q "Total Time" "$log" || VTM_NNINTERP_DIR=$EXPORT VTM_NNINTERP_THREADS=1 $ENC_M -c $base -c $CFG/per-sequence/$pcfg.cfg -i $YUV/$yuv.yuv -q $qp -f $frames $ipopt -b $out/${name}_q$qp.bin --SEIDecodedPictureHash=1 > $log 2>&1
  cmp -s $out/${name}_q$qp.bin $mb && echo "ENCT STREAM SAME $tag" >> $ROOT/crosscheck.txt || echo "ENCT STREAM DIFF $tag" >> $ROOT/crosscheck.txt
  echo "post done $tag $(date '+%m-%d_%H:%M')"
}
export -f one; export ROOT B DEC_A DEC_M ENC_M EXPORT YUV CFG
JOBS=$ROOT/jobs/jobs_post_$(echo $CLASSES | tr ' ' '_').txt; : > $JOBS
for f in $ROOT/jobs/jobs_model_*.txt $ROOT/jobs/jobs_modelgpu_*.txt; do [ -f $f ] && awk -v cl=" $CLASSES " '{ if (index(cl, " "$3" ")) print $1,$2,$3,$4,$5,$6 }' $f; done | sort -u > $JOBS
echo "C2S post [$CLASSES] START $(date '+%m-%d_%H:%M') jobs $(wc -l < $JOBS) par $PAR" >> $ROOT/progress.txt
sort -k2,2n $JOBS | xargs -P $PAR -L 1 bash -c 'one "$@"' _ >> $ROOT/progress.txt 2>&1
echo "C2S post [$CLASSES] DONE $(date '+%m-%d_%H:%M')" >> $ROOT/progress.txt
