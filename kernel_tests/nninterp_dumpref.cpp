// yvvyee: 타일 덤프(4640 int16) 레코드를 가설별로 predict() 에 넣어 VTM 훅의 시험 기록(VTM_NNINTERP_DBG)과 같은 형식으로 쓴다.
// 사용: nninterp_dumpref export_dir tile.dmp out.bin [poc]   (poc 를 주면 그 POC 레코드만)
// 출력 레코드: int32 hdr[16] = {poc, x, y, tw, th, dpoc, fx, fy, st, imv, merge, w4, h4, 0, hyp, 0}, int32 y[256] (Q10.4)
#include "NNInterp.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char **argv)
{
  if (argc < 4) { fprintf(stderr, "usage: %s export_dir tile.dmp out.bin [poc]\n", argv[0]); return 1; }
  NNInterp &nn = NNInterp::get(); if (!nn.load(argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
  FILE *df = fopen(argv[2], "rb"), *of = fopen(argv[3], "wb"); if (!df || !of) { fprintf(stderr, "open fail\n"); return 1; }
  const int wantPoc = argc > 4 ? atoi(argv[4]) : -1;
  std::vector<int16_t> r(4640); long n = 0, w = 0;
  while (fread(r.data(), 2, 4640, df) == 4640)
  {
    n++;
    const int st = r[0], qp = r[1], poc = r[2], px = r[3], py = r[4], pw = r[5], ph = r[6], tx = r[7], ty = r[8], tw = r[9], th = r[10];
    const int imv = r[17], merge = r[18], nh = r[21];
    if (wantPoc >= 0 && poc != wantPoc) continue;
    for (int h = 0; h < nh; h++)
    {
      NNInterp::BlockIn in; NNInterp::BlockOut out;
      in.tw = tw; in.th = th; in.stride = tw + 16; in.fx = r[11 + 2 * h]; in.fy = r[12 + 2 * h]; in.qp = qp; in.imv = imv; in.merge = merge;
      in.dpoc = r[19 + h]; in.w4 = pw / 4; in.h4 = ph / 4; in.st = st;
      for (int y = 0; y < th + 16; y++)
        for (int x = 0; x < tw + 16; x++)
        {
          in.win[y * (tw + 16) + x] = r[32 + h * 1024 + y * 32 + x];
          in.map[y * (tw + 16) + x] = (uint16_t) r[32 + 2048 + h * 1024 + y * 32 + x];
        }
      nn.predict(in, out);
      int32_t hdr[16] = { poc, px + tx, py + ty, tw, th, in.dpoc, in.fx, in.fy, st, imv, merge, in.w4, in.h4, 0, h, 0 };
      fwrite(hdr, 4, 16, of);
      int32_t yv[256] = { 0 }; for (int k = 0; k < tw * th; k++) yv[k] = (int32_t) out.y[k];
      fwrite(yv, 4, 256, of); w++;
    }
  }
  fclose(of); printf("records %ld written %ld\n", n, w); return 0;
}
