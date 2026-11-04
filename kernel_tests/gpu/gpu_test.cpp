// yvvyee: NNInterpCuda test harness.
//  gpu_test export_dir tile.dmp ref.i32 [fuzzN=3000] [seeds=3]
//  (a) tile rows vs reference file (same comparison as nninterp_tile_test.cpp: y, ylin, xd, final average)
//  (b) random BlockIn cases (same generator as ks/fuzz.cpp) vs NNInterp::predictRef on the CPU
//  (c) throughput (pinned host buffers, H2D + kernel + D2H + sync per call) and per-call latency
#include "nninterp_cuda.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <random>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

typedef NNInterp::BlockIn BI; typedef NNInterp::BlockOut BO;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv)
{
  if (argc < 4) { fprintf(stderr, "usage: %s export_dir tile.dmp ref.i32 [fuzzN] [seeds] [skip: a|b|c letters to run]\n", argv[0]); return 1; }
  const int fuzzN = argc > 4 ? atoi(argv[4]) : 3000, nSeeds = argc > 5 ? atoi(argv[5]) : 3; const std::string run = argc > 6 ? argv[6] : "abc";
  NNInterp &cpu = NNInterp::get(); if (!cpu.load(argv[1])) { fprintf(stderr, "cpu load fail\n"); return 1; }
  NNInterpCuda gpu; if (!gpu.load(argv[1])) { fprintf(stderr, "gpu load fail: %s\n", gpu.lastError()); return 1; }
  int rc = 0;
  const int CH = 4096; BI *pin = (BI *) NNInterpCuda::allocPinned(sizeof(BI) * CH); BO *pout = (BO *) NNInterpCuda::allocPinned(sizeof(BO) * CH);
  if (!pin || !pout) { fprintf(stderr, "pinned alloc fail\n"); return 1; }

  // ---------------- (a) tile rows
  std::vector<BI> pool16, poolAll;   // for (c)
  {
    FILE *df = fopen(argv[2], "rb"), *rf = fopen(argv[3], "rb"); if (!df || !rf) { fprintf(stderr, "open fail\n"); return 1; }
    const int REC = 4640, ROW = 1940; std::vector<int32_t> rows; { std::vector<int32_t> r(ROW); while (fread(r.data(), 4, ROW, rf) == (size_t) ROW) rows.insert(rows.end(), r.begin(), r.end()); }
    const long N = (long) rows.size() / ROW; std::vector<int16_t> rec(REC);
    std::vector<BI> ins(N);
    for (long i = 0; i < N; i++)
    {
      const int32_t *row = &rows[i * ROW]; const int r = row[0], hyp = row[1], tw = row[2], th = row[3];
      fseek(df, (long) r * REC * 2, SEEK_SET); if (fread(rec.data(), 2, REC, df) != (size_t) REC) { fprintf(stderr, "dump read fail\n"); return 1; }
      const int16_t *H = rec.data(); BI &in = ins[i];
      in.stride = 32; in.tw = tw; in.th = th;
      for (int k = 0; k < 1024; k++) { in.win[k] = rec[32 + hyp * 1024 + k]; in.map[k] = rec[2080 + hyp * 1024 + k]; }
      in.fx = std::max<int>(0, H[11 + 2 * hyp]); in.fy = std::max<int>(0, H[12 + 2 * hyp]); in.qp = H[1]; in.imv = H[17]; in.merge = H[18]; in.dpoc = H[19 + hyp];
      in.w4 = H[5] >> 2; in.h4 = H[6] >> 2; in.st = H[0];
    }
    if (run.find('a') != std::string::npos)
    {
      long okY = 0, okL = 0, okX = 0, okF = 0, nF = 0; std::map<int, std::vector<int64_t>> y0s; double t0 = now();
      for (long b = 0; b < N; b += CH)
      {
        const int n = (int) std::min<long>(CH, N - b); memcpy(pin, &ins[b], sizeof(BI) * n);
        if (!gpu.predictBatch(pin, pout, n, NNInterpCuda::OUT_ALL)) { fprintf(stderr, "predictBatch: %s\n", gpu.lastError()); return 1; }
        for (int j = 0; j < n; j++)
        {
          const int32_t *row = &rows[(b + j) * ROW]; const int r = row[0], hyp = row[1], tw = row[2], th = row[3]; const BO &o = pout[j];
          const int nY = th * tw, nL = (th + 4) * (tw + 4), nX = (th + 16) * (tw + 16); bool eY = true, eL = true, eX = true;
          for (int k = 0; k < nY; k++) eY &= o.y[k] == row[4 + k];
          for (int k = 0; k < nL; k++) eL &= o.ylin[k] == row[260 + k];
          for (int k = 0; k < nX; k++) eX &= o.xd[k] == row[660 + k];
          okY += eY; okL += eL; okX += eX;
          fseek(df, (long) r * REC * 2, SEEK_SET); int16_t H0; if (fread(&H0, 2, 1, df) != 1) return 1;
          if (H0 == 0 || hyp == 1)
          {
            bool eF = true;
            for (int k = 0; k < nY; k++) { const int64_t f = (H0 == 0) ? o.y[k] : ((y0s[r][k] + o.y[k] + 1) >> 1); eF &= f == row[1684 + k]; }
            nF++; okF += eF; y0s.erase(r);
          }
          else y0s[r].assign(o.y, o.y + nY);
        }
      }
      printf("[a] tile rows %ld | y exact %ld | ylin exact %ld | xd exact %ld | final exact %ld/%ld | pending %zu | %.2f s\n", N, okY, okL, okX, okF, nF, y0s.size(), now() - t0);
      if (okY != N || okL != N || okX != N || okF != nF) rc = 1;
    }
    for (long i = 0; i < N; i++) { poolAll.push_back(ins[i]); if (ins[i].tw == 16 && ins[i].th == 16) pool16.push_back(ins[i]); }
    fclose(df); fclose(rf);
  }

  // ---------------- (b) fuzz vs predictRef
  if (run.find('b') != std::string::npos)
    for (int seed = 0; seed < nSeeds; seed++)
    {
      std::mt19937_64 rng(seed); auto U = [&](int a, int b) { return (int) (a + rng() % (uint64_t) (b - a + 1)); };
      std::vector<BI> cases(fuzzN); std::vector<int> modes(fuzzN);
      for (int it = 0; it < fuzzN; it++)
      {
        BI &in = cases[it];
        in.tw = 4 * U(1, 4); in.th = 4 * U(1, 4); const int Ww = in.tw + 16, Hh = in.th + 16; in.stride = U(Ww, 1024 / Hh > 32 ? 32 : Ww);
        if (in.stride * (Hh - 1) + Ww > 1024) in.stride = Ww;
        const int mode = U(0, 7); modes[it] = mode; int lo = 0, hi = 1023;
        if (mode == 5) { lo = -4096; hi = 8191; }
        for (int i = 0; i < 1024; i++)
        {
          const int yy = i / in.stride, xx = i % in.stride; int v;
          switch (mode)
          {
          case 0: v = U(lo, hi); break;
          case 1: v = ((yy + xx) & 1) ? hi : lo; break;
          case 2: v = U(0, 1) ? hi : lo; break;
          case 3: v = 512; break;
          case 4: v = (xx * 37 + yy * 11) % 1024; break;
          case 5: v = U(lo, hi); break;
          case 6: v = (yy & 2) ? 1023 : 0; break;
          default: v = std::min(1023, std::max(0, 512 + (int) ((rng() % 201) - 100) * (xx % 3))); break;
          }
          in.win[i] = v; in.map[i] = (int) (rng() & 0x7fff);
        }
        in.fx = U(-1, 15); in.fy = U(-1, 15); if (U(0, 3) == 0) { in.fx = 8 * U(0, 1); in.fy = 8 * U(0, 1); }
        in.qp = U(0, 63); in.imv = U(0, 3); if (U(0, 2) == 0) in.imv = 3; in.merge = U(0, 1); in.dpoc = U(-32, 32); in.w4 = U(1, 32); in.h4 = U(1, 32); in.st = U(0, 2);
      }
      std::vector<BO> ref(fuzzN); double t0 = now();
#pragma omp parallel for schedule(dynamic)
      for (int it = 0; it < fuzzN; it++) cpu.predictRef(cases[it], ref[it]);
      const double tc = now() - t0; long bad = 0;
      for (int b = 0; b < fuzzN; b += CH)
      {
        const int n = std::min(CH, fuzzN - b); memcpy(pin, &cases[b], sizeof(BI) * n);
        if (!gpu.predictBatch(pin, pout, n, NNInterpCuda::OUT_ALL)) { fprintf(stderr, "predictBatch: %s\n", gpu.lastError()); return 1; }
        for (int j = 0; j < n; j++)
        {
          const BI &in = cases[b + j]; const BO &o1 = pout[j], &o2 = ref[b + j];
          const int nY = in.tw * in.th, nL = (in.tw + 4) * (in.th + 4), nX = (in.tw + 16) * (in.th + 16);
          const bool ok = !memcmp(o1.y, o2.y, 8 * nY) && !memcmp(o1.ylin, o2.ylin, 8 * nL) && !memcmp(o1.xd, o2.xd, 8 * nX);
          if (!ok)
          {
            bad++;
            if (bad < 6)
            {
              int fy = -1, fl = -1, fxd = -1; for (int k = nY - 1; k >= 0; k--) if (o1.y[k] != o2.y[k]) fy = k; for (int k = nL - 1; k >= 0; k--) if (o1.ylin[k] != o2.ylin[k]) fl = k; for (int k = nX - 1; k >= 0; k--) if (o1.xd[k] != o2.xd[k]) fxd = k;
              fprintf(stderr, "mismatch seed %d it %d mode %d tile %dx%d fx %d fy %d imv %d | first y %d ylin %d xd %d\n", seed, b + j, modes[b + j], in.tw, in.th, in.fx, in.fy, in.imv, fy, fl, fxd);
            }
          }
        }
      }
      printf("[b] fuzz seed %d: %d cases, mismatches %ld (cpu predictRef %.1f s on %d threads)\n", seed, fuzzN, bad, tc, omp_get_max_threads());
      if (bad) rc = 1;
    }

  // ---------------- (c) throughput and latency
  if (run.find('c') != std::string::npos)
  {
    const std::vector<BI> &pool = pool16.empty() ? poolAll : pool16;
    printf("[c] throughput, %s tiles from tile1.dmp (pool %zu), pinned host buffers, per call: H2D + kernel + D2H + sync\n", pool16.empty() ? "mixed" : "16x16", pool.size());
    for (size_t i = 0; i < (size_t) CH; i++) pin[i] = pool[i % pool.size()];
    const int Bs[6] = { 1, 4, 16, 64, 256, 1024 };
    printf("%6s | %14s %14s | %14s %14s | %12s %12s\n", "batch", "us/tile(Y)", "us/call(Y)", "us/tile(ALL)", "us/call(ALL)", "p50 call(Y)", "p99 call(Y)");
    for (int B : Bs)
    {
      double res[2][2]; std::vector<double> lat;
      for (int mode = 0; mode < 2; mode++)
      {
        const int mask = mode == 0 ? NNInterpCuda::OUT_Y : NNInterpCuda::OUT_ALL;
        for (int w = 0; w < 3; w++) gpu.predictBatch(pin, pout, B, mask);   // warm-up (buffer growth)
        int iters = 0; const double t0 = now(); double t = t0;
        while (t - t0 < 1.0 || iters < 20)
        {
          const double a = now(); if (!gpu.predictBatch(pin + (iters * B) % (CH - B + 1), pout, B, mask)) { fprintf(stderr, "%s\n", gpu.lastError()); return 1; }
          t = now(); if (mode == 0) lat.push_back(t - a); iters++;
        }
        res[mode][0] = 1e6 * (t - t0) / ((double) iters * B); res[mode][1] = 1e6 * (t - t0) / iters;
      }
      std::sort(lat.begin(), lat.end());
      printf("%6d | %14.3f %14.1f | %14.3f %14.1f | %12.1f %12.1f\n", B, res[0][0], res[0][1], res[1][0], res[1][1], 1e6 * lat[lat.size() / 2], 1e6 * lat[lat.size() * 99 / 100]);
    }
    // yvvyee: large batch for the throughput ceiling
    {
      const int B = CH; for (int w = 0; w < 2; w++) gpu.predictBatch(pin, pout, B, NNInterpCuda::OUT_Y);
      int iters = 0; const double t0 = now(); while (now() - t0 < 2.0) { gpu.predictBatch(pin, pout, B, NNInterpCuda::OUT_Y); iters++; }
      printf("%6d | %14.3f %14.1f |  (OUT_Y only)\n", B, 1e6 * (now() - t0) / ((double) iters * B), 1e6 * (now() - t0) / iters);
    }
    size_t fr, tot; cudaMemGetInfo(&fr, &tot);
    printf("[c] device bytes held by NNInterpCuda (params + batch buffers at cap): %.1f MiB; device used total (all processes): %.1f MiB\n", gpu.deviceBytes() / 1048576.0, (tot - fr) / 1048576.0);
  }
  NNInterpCuda::freePinned(pin); NNInterpCuda::freePinned(pout);
  return rc;
}
