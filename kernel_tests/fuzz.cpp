// yvvyee: predict (가속) 와 predictRef (기준) 의 임의 입력 비트 비교
#include "NNInterp.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
int main(int argc, char **argv)
{
  NNInterp &nn = NNInterp::get(); if (!nn.load(argv[1])) return 1;
  const int N = atoi(argv[2]); std::mt19937_64 rng(atoi(argv[3]));
  auto U = [&](int a, int b) { return (int) (a + rng() % (uint64_t) (b - a + 1)); };
  NNInterp::BlockIn in; static NNInterp::BlockOut o1, o2; long bad = 0; const bool lin = getenv("NNI_LINEAR") != nullptr;   // yvvyee: proxy mode: predict(linearOnly) y must equal the y_lin tile area of predictRef

  for (int it = 0; it < N; it++)
  {
    in.tw = 4 * U(1, 4); in.th = 4 * U(1, 4); const int Ww = in.tw + 16, Hh = in.th + 16; in.stride = U(Ww, 1024 / Hh > 32 ? 32 : Ww);
    if (in.stride * (Hh - 1) + Ww > 1024) in.stride = Ww;
    const int mode = U(0, 7); int lo = 0, hi = 1023;
    if (mode == 5) { lo = -4096; hi = 8191; }   // 범위 밖 화소 (가드·대체 경로 시험)
    for (int i = 0; i < 1024; i++)
    {
      const int yy = i / in.stride, xx = i % in.stride; int v;
      switch (mode)
      {
      case 0: v = U(lo, hi); break;
      case 1: v = ((yy + xx) & 1) ? hi : lo; break;                       // 바둑판 (최대 고주파)
      case 2: v = U(0, 1) ? hi : lo; break;                               // 무작위 극값
      case 3: v = 512; break;                                             // 평탄
      case 4: v = (xx * 37 + yy * 11) % 1024; break;                      // 기울기 + 접힘
      case 5: v = U(lo, hi); break;
      case 6: v = (yy & 2) ? 1023 : 0; break;                              // 가로 줄무늬
      default: v = std::min(1023, std::max(0, 512 + (int) ((rng() % 201) - 100) * (xx % 3))); break;
      }
      in.win[i] = v; in.map[i] = (int) (rng() & 0x7fff);
    }
    in.fx = U(-1, 15); in.fy = U(-1, 15); if (U(0, 3) == 0) { in.fx = 8 * U(0, 1); in.fy = 8 * U(0, 1); }
    in.qp = U(0, 63); in.imv = U(0, 3); if (U(0, 2) == 0) in.imv = 3; in.merge = U(0, 1); in.dpoc = U(-32, 32); in.w4 = U(1, 32); in.h4 = U(1, 32); in.st = U(0, 2);
    nn.predict(in, o1, lin); nn.predictRef(in, o2);
    if (lin) for (int yy = 0; yy < in.th; yy++) for (int xx = 0; xx < in.tw; xx++) o2.y[yy * in.tw + xx] = o2.ylin[(yy + 2) * (in.tw + 4) + xx + 2];
    const int nY = in.tw * in.th, nL = (in.tw + 4) * (in.th + 4), nX = Ww * Hh;
    const bool ok = !memcmp(o1.y, o2.y, 8 * nY) && !memcmp(o1.ylin, o2.ylin, 8 * nL) && !memcmp(o1.xd, o2.xd, 8 * nX);
    if (!ok) { bad++; if (bad < 5) fprintf(stderr, "mismatch it %d mode %d tile %dx%d fx %d fy %d imv %d\n", it, mode, in.tw, in.th, in.fx, in.fy, in.imv); }
  }
  printf("%sfuzz %d cases", lin ? "[linearOnly] " : "", N); printf(", mismatches %ld\n", bad); return bad != 0;
}
