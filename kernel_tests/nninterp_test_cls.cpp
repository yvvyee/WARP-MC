// yvvyee: NNInterp 비트 일치 시험기 — 내보내기 기준 출력과 비교. 빌드: g++ -O2 -std=c++17 -I<VTM CommonLib> nninterp_test.cpp <CommonLib>/NNInterp.cpp
#include "NNInterp.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
int main(int argc, char **argv)
{
  if (argc < 3) { fprintf(stderr, "usage: %s export_dir dump.dmp\n", argv[0]); return 1; }
  NNInterp &nn = NNInterp::get(); if (!nn.load(argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
  FILE *df = fopen(argv[2], "rb"); const int REC = 1936; std::vector<int16_t> rec(REC);
  const auto &ri = nn.A_("ref_idx"), &rxd = nn.A_("ref_xd_q4"), &ryl = nn.A_("ref_ylin_q4"), &ry = nn.A_("ref_y_q4"); const int nref = (int) ri.shape[0];
  long okx = 0, okl = 0, oky = 0; long long mdx = 0, mdl = 0, mdy = 0; int firstBad = -1;
  for (int k = 0; k < nref; k++)
  {
    fseek(df, (long) ri.v[k] * REC * 2, SEEK_SET); if (fread(rec.data(), 2, REC, df) != (size_t) REC) break;
    NNInterp::BlockIn in; NNInterp::BlockOut o;
    for (int i = 0; i < 400; i++) { in.win[i] = rec[330 + i]; in.map[i] = rec[1136 + i]; }
    in.fx = std::max<int>(0, rec[5]); in.fy = std::max<int>(0, rec[6]); in.qp = rec[1]; in.imv = rec[1130]; in.merge = rec[1131]; in.dpoc = rec[1133]; in.w4 = (rec[1135] >> 8) & 255; in.h4 = rec[1135] & 255; in.st = rec[0];
    nn.predict(in, o);
    long long dx = 0, dl = 0, dy = 0;
    for (int i = 0; i < 400; i++) dx = std::max(dx, (long long) std::abs(o.xd[i] - rxd.v[(int64_t) k * 400 + i]));
    for (int i = 0; i < 64; i++) dl = std::max(dl, (long long) std::abs(o.ylin[i] - ryl.v[(int64_t) k * 64 + i]));
    for (int i = 0; i < 16; i++) dy = std::max(dy, (long long) std::abs(o.y[i] - ry.v[(int64_t) k * 16 + i]));
    okx += (dx == 0); okl += (dl == 0); oky += (dy == 0); mdx = std::max(mdx, dx); mdl = std::max(mdl, dl); mdy = std::max(mdy, dy); if (dy != 0 && firstBad < 0) firstBad = k;
  }
  printf("ref %d | xd exact %ld (max|d| %lld) | y_lin exact %ld (max|d| %lld) | y exact %ld (max|d| %lld) | first y mismatch %d\n", nref, okx, mdx, okl, mdl, oky, mdy, firstBad);
  fclose(df); return 0;
}
