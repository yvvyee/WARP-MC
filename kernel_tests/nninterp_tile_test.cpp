// yvvyee: NNInterp 타일 비트 일치 시험기 — 타일 덤프(4640 int16 레코드)를 행 간격 32 창으로 predict() 에 넣고,
// 모사기(qkv_fixed.py --tile ... ref_out --f64)가 쓴 레코드·가설별 기준 출력과 비교한다. (tw,th) 그룹별 일치 수와 predict() 시간을 낸다.
// 빌드: g++ -O3 -march=native -std=c++17 -I<VTM CommonLib> nninterp_tile_test.cpp <CommonLib>/NNInterp.cpp
// 기준 행 (int32 1940) = [rec, hyp, tw, th] + y[256] + ylin[400] + xd[1024] + yfin[256] (조밀 행 우선)
#include "NNInterp.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <vector>
int main(int argc, char **argv)
{
  if (argc < 4) { fprintf(stderr, "usage: %s export_dir tile.dmp ref.i32\n", argv[0]); return 1; }
  NNInterp &nn = NNInterp::get(); if (!nn.load(argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
  FILE *df = fopen(argv[2], "rb"), *rf = fopen(argv[3], "rb"); if (!df || !rf) { fprintf(stderr, "open fail\n"); return 1; }
  const int REC = 4640, ROW = 1940; std::vector<int16_t> rec(REC); std::vector<int32_t> row(ROW);
  struct G { long rows = 0, okY = 0, okL = 0, okX = 0, fin = 0, okF = 0; long long mdY = 0, mdL = 0, mdX = 0, mdF = 0; double sec = 0; };
  std::map<std::pair<int, int>, G> grp; std::map<int, std::vector<int64_t>> y0s;   // 양방향: 가설 0 출력 보관
  NNInterp::BlockIn in; NNInterp::BlockOut o;
  const bool lin = getenv("NNI_LINEAR") != nullptr;   // yvvyee: proxy mode: expected y = tile area of the reference y_lin
  while (fread(row.data(), 4, ROW, rf) == (size_t) ROW)
  {
    const int r = row[0], hyp = row[1], tw = row[2], th = row[3];
    fseek(df, (long) r * REC * 2, SEEK_SET); if (fread(rec.data(), 2, REC, df) != (size_t) REC) { fprintf(stderr, "dump read fail %d\n", r); return 1; }
    const int16_t *H = rec.data();
    if (H[9] != tw || H[10] != th) { fprintf(stderr, "tile size mismatch rec %d\n", r); return 1; }
    in.stride = 32; in.tw = tw; in.th = th;
    for (int i = 0; i < 1024; i++) { in.win[i] = rec[32 + hyp * 1024 + i]; in.map[i] = rec[2080 + hyp * 1024 + i]; }
    in.fx = std::max<int>(0, H[11 + 2 * hyp]); in.fy = std::max<int>(0, H[12 + 2 * hyp]); in.qp = H[1]; in.imv = H[17]; in.merge = H[18]; in.dpoc = H[19 + hyp];
    in.w4 = H[5] >> 2; in.h4 = H[6] >> 2; in.st = H[0];
    const auto t0 = std::chrono::steady_clock::now(); nn.predict(in, o, lin); const auto t1 = std::chrono::steady_clock::now();
    G &g = grp[{ tw, th }]; g.rows++; g.sec += std::chrono::duration<double>(t1 - t0).count();
    const int nY = th * tw, nL = (th + 4) * (tw + 4), nX = (th + 16) * (tw + 16);
    long long dY = 0, dL = 0, dX = 0;
    for (int i = 0; i < nY; i++) dY = std::max(dY, std::llabs(o.y[i] - (lin ? row[260 + (i / tw + 2) * (tw + 4) + i % tw + 2] : row[4 + i])));
    for (int i = 0; i < nL; i++) dL = std::max(dL, std::llabs(o.ylin[i] - row[260 + i]));
    for (int i = 0; i < nX; i++) dX = std::max(dX, std::llabs(o.xd[i] - row[660 + i]));
    if (dX != 0 && getenv("NNI_VERBOSE"))   // yvvyee: xd 불일치 위치 출력
      for (int i = 0; i < nX; i++) if (o.xd[i] != row[660 + i]) printf("xd mismatch rec %d hyp %d tile %dx%d at (y %d, x %d): cpp %lld ref %d\n", r, hyp, tw, th, i / (tw + 16), i % (tw + 16), (long long) o.xd[i], row[660 + i]);
    g.okY += (dY == 0); g.okL += (dL == 0); g.okX += (dX == 0); g.mdY = std::max(g.mdY, dY); g.mdL = std::max(g.mdL, dL); g.mdX = std::max(g.mdX, dX);
    // 최종 출력: 단방향 = 가설 0, 양방향 = Q10.4 평균 floor((y0+y1)/2 + 0.5)
    if (!lin && (H[0] == 0 || hyp == 1))   // yvvyee: final (bi average of y) is checked in full mode only
    {
      long long dF = 0;
      for (int i = 0; i < nY; i++) { const int64_t f = (H[0] == 0) ? o.y[i] : ((y0s[r][i] + o.y[i] + 1) >> 1); dF = std::max(dF, std::llabs(f - row[1684 + i])); }
      g.fin++; g.okF += (dF == 0); g.mdF = std::max(g.mdF, dF); y0s.erase(r);
    }
    else y0s[r].assign(o.y, o.y + nY);
  }
  printf("%-6s %7s | %-22s %-22s %-22s | %-22s | %9s %12s\n", "tile", "rows", "y exact (max|d|)", "y_lin exact (max|d|)", "xd exact (max|d|)", "final exact (max|d|)", "ms/tile", "us/out-pixel");
  long T = 0, TY = 0, TL = 0, TX = 0, TF = 0, TFn = 0;
  for (auto &kv : grp)
  {
    const G &g = kv.second; const int tw = kv.first.first, th = kv.first.second; char b[4][64];
    snprintf(b[0], 64, "%ld/%ld (%lld)", g.okY, g.rows, g.mdY); snprintf(b[1], 64, "%ld/%ld (%lld)", g.okL, g.rows, g.mdL); snprintf(b[2], 64, "%ld/%ld (%lld)", g.okX, g.rows, g.mdX); snprintf(b[3], 64, "%ld/%ld (%lld)", g.okF, g.fin, g.mdF);
    printf("%2dx%-3d %7ld | %-22s %-22s %-22s | %-22s | %9.3f %12.3f\n", tw, th, g.rows, b[0], b[1], b[2], b[3], 1e3 * g.sec / g.rows, 1e6 * g.sec / g.rows / (tw * th));
    T += g.rows; TY += g.okY; TL += g.okL; TX += g.okX; TF += g.okF; TFn += g.fin;
  }
  if (lin) printf("[linearOnly] y compared with the y_lin tile crop, final not applicable\n");
  printf("total rows %ld | y exact %ld | y_lin exact %ld | xd exact %ld | final exact %ld/%ld | pending bi %zu\n", T, TY, TL, TX, TF, TFn, y0s.size());
  fclose(df); fclose(rf); return 0;
}
