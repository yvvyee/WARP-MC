/* yvvyee: CUDA port of NNInterp::predictRef (VTM-23.8_nn CommonLib/NNInterp.cpp), batched over tiles.
 * yvvyee: design rules for bit-exactness
 *  - Every stage keeps the reference operation order, rounding (floor(x/2^s + 1/2)), saturation points and table lookups.
 *  - Where the reference accumulates in __int128, the device code also uses __int128 (nvcc supports it in device code since 11.5),
 *    except where the CPU fast path already proves int64 is exact; those proofs are repeated here (load-time on parameters, or a
 *    block-uniform runtime flag via __syncthreads_or on data bounds, falling back to the __int128 expression otherwise).
 *  - L7 (3x3 convs, FiltAct) uses int32 only under the same load-time L1-norm bounds the CPU fast path checks (load() fails otherwise).
 *  - The reference's int64 rshr with a negative shift is a left shift; on x86 the count is taken mod 64, so the device version masks too.
 * yvvyee: mapping: one thread block (NT threads) per tile, all intermediates in one dynamic shared-memory arena reused across stages.
 */
#include "NNInterpCuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <map>
#include <vector>

typedef __int128 i128;

namespace {

constexpr int NT = 512;        // yvvyee: threads per tile block
constexpr int KMAX = 25;       // yvvyee: candidate count upper bound (keeps per-pixel attention arrays in registers)
constexpr int NHMAX = 5;       // yvvyee: distinct horizontal candidate shifts (incl. 0) held in shared memory
constexpr int PIXF = 4, FIRF = 14;

struct C1D { int Co, Ci, we, oexp; };

struct DevP
{
  int firN[4]; int firT[4][32];                 // gauss06, gauss12, gauss24, gauss10
  int k12[16 * 12]; int k7[7]; long long gate;
  int K, nh, hsx[NHMAX], hid[KMAX], cdy[KMAX]; int phiM[KMAX * 5];
  int eSq[4], ePhi[4], ePhin, eSt0, eSt, ephi, eD, eIn, sc4;
  long long LOG2E, LN2, INV30;
  long long tabB[32], rth[64], r15th[64]; int eth[64];
  int D[16];
  C1D n0, n2, q0, q2;
  int n0w[32 * 28], n2w[8 * 32], q0w[32 * 24], q2w[8 * 32];
  long long n0bo[32], n2bo[8], q0bo[32], q2bo[8];   // bias already shifted to the accumulator scale
  int n0sh, n2sh, q0sh, q2sh, q0up[24];
  int c3Co[5], c3Ci[5], c3off[5], c3sh[5]; long long c3bo[5][32];
  int faA[4], faB[4], faD[4], faE[4]; int hu2[12], hd[12];
  int lutExp, lutLog, lutRec, lutSig;           // offsets into the LUT array
};
__constant__ DevP P;

#ifdef NNI_PROF
__device__ long long g_prof[64];   // yvvyee: per-stage clock64 stamps of block 0 (profiling build only)
#define PROF(k) do { __syncthreads(); if (threadIdx.x == 0 && blockIdx.x == 0) g_prof[k] = clock64(); } while (0)
#else
#define PROF(k) do { } while (0)
#endif

// ---------------------------------------------------------------- arena layout (bytes)
constexpr int OFF_YLIN = 0;                       // int64[400]
constexpr int OFF_SMALL = 3200;                   // sd int64[16], qtab int64[64], baseA int64[32], baseQ int64[32]
constexpr int OFF_XD = 4352;                      // int64[1024]
constexpr int OFF_LPN = OFF_XD + 8192;            // int16[4][1024]
constexpr int OFF_CF = OFF_LPN + 8192;            // int16[12][1024]
constexpr int OFF_S = OFF_CF + 24576;             // 45312: phase scratch
// phase A (relative to OFF_S)
constexpr int A_XB3 = 0, A_G3 = 8192, A_TMP = 14336, A_SQ = 18432, A_PHI = 22528, A_END = 38912;   // CM aliases PHI
// phase B (relative to OFF_S)
constexpr int B_XH = 0, B_S0 = 0, B_TMP = 4096, B_S1V = 8192;   // lst FIR temps alias XH (dead before XH is written)
constexpr int B_LST = NHMAX * 32 * 28 * 8, B_Q = B_LST + 2400, B_END = B_Q + 6400;                // CIN0 aliases LST
constexpr int ARENA = OFF_S + (A_END > B_END ? A_END : B_END);   // front kernel arena
// yvvyee: L7 kernels run one block per (tile, OPB output channels) so that a small batch still spreads over many SMs. FiltAct is
// per channel, so a block applies it to its own conv outputs without exchanging data. Activations between layers live in global
// memory as packed channel pairs (PKW words per tile), which stays L2-resident.
constexpr int OPB = 2, NTC = 256, PLANE = 784, PKW = 8 * 900;   // yvvyee: PKW = packed words per tile in global (8 channel pairs x (Cy+2)(Cx+2))
constexpr int L_PK = 0, L_WK = 28800, L_PL = L_WK + 9 * 8 * 4 * 4, L_END = L_PL + 4 * PLANE * 2;   // PK uint32[8][900], WK, planes
constexpr int L_C4 = L_PL, L_CC = L_PL + 1024, L8_END = L_CC + 4096;                                // conv4 + L8 kernel
static_assert(2 * OPB * 28 * 56 * 2 + (NTC / 32) * 64 * 4 <= L_WK, "FiltAct temps must fit in the PK area");

// ---------------------------------------------------------------- arithmetic (same definitions as NNInterp.cpp)
__device__ __forceinline__ long long rshrW(i128 x, int sh)   // reference rshr(i128, sh)
{
  if (sh <= 0) return (long long) (x << (-sh));
  if (sh >= 120) return 0;
  return (long long) ((x + ((i128) 1 << (sh - 1))) >> sh);
}
__device__ __forceinline__ long long rshrL(long long x, int sh)   // reference rshr(int64, sh); x86 shl masks the count
{
  if (sh <= 0) return (long long) ((unsigned long long) x << ((-sh) & 63));
  if (sh >= 63) return 0;
  return (x + (1LL << (sh - 1))) >> sh;
}
__device__ __forceinline__ long long rdivPow2(i128 num, int k) { return (long long) ((num + ((i128) 1 << (k - 1))) >> k); }   // fdiv_round(num, 2^k), k >= 1
__device__ __forceinline__ long long fdivRoundL(long long num, long long den)   // fdiv_round for values that fit int64
{
  const long long n2 = 2 * num + den, d2 = 2 * den;
  return n2 >= 0 ? n2 / d2 : -((-n2 + d2 - 1) / d2);
}
__device__ __forceinline__ int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
__device__ __forceinline__ long long clamp16(long long x) { return x < -32767 ? -32767 : (x > 32767 ? 32767 : x); }
__device__ __forceinline__ long long clamp32(long long x) { return x < -2147483647LL ? -2147483647LL : (x > 2147483647LL ? 2147483647LL : x); }
__device__ __forceinline__ int clamp16i(int x) { return x < -32767 ? -32767 : (x > 32767 ? 32767 : x); }
__device__ __forceinline__ int refl(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); }
__device__ __forceinline__ int msbW(i128 x)
{
  if (x <= 0) return -1;
  const unsigned long long hi = (unsigned long long) ((unsigned __int128) x >> 64), lo = (unsigned long long) x;
  return hi ? 127 - __clzll((long long) hi) : 63 - __clzll((long long) lo);
}
__device__ __forceinline__ void normalizeW(i128 num, int sc, int &e, int &m10)
{
  const int msb = msbW(num); e = msb - sc;
  m10 = msb >= 1 ? (int) (((num << 10) + ((i128) 1 << (msb - 1))) >> msb) : 1024;
}
struct Lut { const int *t; };
__device__ __forceinline__ void expQ(const int *lut, long long x8, long long &t, int &n)
{
  long long t8 = rshrW((i128) x8 * P.LOG2E, 16); t8 = t8 < -86 * 256 ? -86 * 256 : (t8 > 86 * 256 ? 86 * 256 : t8);
  n = (int) (t8 >> 8); t = __ldg(lut + P.lutExp + (int) (t8 - 256LL * n));
}
__device__ __forceinline__ long long lnQ10(const int *lut, i128 num, int sc)
{
  int e, m10; normalizeW(num, sc, e, m10); const long long l2 = (long long) e * 1024 + __ldg(lut + P.lutLog + m10 - 1024);
  return rshrW((i128) l2 * P.LN2, 16);
}
__device__ __forceinline__ void recipQ(const int *lut, i128 num, int sc, long long &r15, int &e) { int m10; normalizeW(num, sc, e, m10); r15 = __ldg(lut + P.lutRec + m10 - 1024); }
__device__ __forceinline__ long long sigQ15(const int *lut, long long x8) { x8 = x8 < -4096 ? -4096 : (x8 > 4096 ? 4096 : x8); return __ldg(lut + P.lutSig + (int) x8 + 4096); }
// energyQ: (a*b) / (PIX_DEN^2 * 2^lgX) at grid 2^e; the denominator is a power of two, so fdiv_round == rdivPow2
__device__ __forceinline__ long long energyQ(long long a, long long b, int e, int lgX)
{
  i128 num = (i128) a * b; if (e >= 0) num <<= e;
  return clamp32(rdivPow2(num, 28 + lgX + (e < 0 ? -e : 0)));
}

// yvvyee: separable FIR with edge replication (sepFirRep). WIDE: __int128 accumulator (reference); otherwise int64, exact when
// |x| < 2^31, sum|h| < 2^24 and both shifts in [1, 62] (checked at load; |x| bound by the caller's block flag).
template <bool WIDE, typename TI, typename TO>
__device__ void sepFir(const TI *x, int H, int W, int f, int *t, TO *y, int s1, int s2, long long lim)
{
  const int T = P.firN[f], r = (T - 1) / 2; const int *h = P.firT[f];
  for (int p = threadIdx.x; p < H * W; p += NT)
  {
    const int yy = p / W, xx = p - yy * W; const TI *xr = x + yy * W; long long v;
    if (WIDE) { i128 acc = 0; for (int k = 0; k < T; k++) acc += (i128) xr[clampi(xx + k - r, 0, W - 1)] * h[k]; v = rshrW(acc, s1); }
    else { long long acc = 0; for (int k = 0; k < T; k++) acc += (long long) (int) xr[clampi(xx + k - r, 0, W - 1)] * h[k]; v = rshrL(acc, s1); }
    t[p] = (int) (v < -lim ? -lim : (v > lim ? lim : v));
  }
  __syncthreads();
  for (int p = threadIdx.x; p < H * W; p += NT)
  {
    const int yy = p / W, xx = p - yy * W; long long acc = 0;
    for (int k = 0; k < T; k++) acc += (long long) t[clampi(yy + k - r, 0, H - 1) * W + xx] * h[k];
    const long long v = rshrL(acc, s2); y[p] = (TO) (v < -lim ? -lim : (v > lim ? lim : v));
  }
  __syncthreads();
}

// yvvyee: dp2a with mixed signedness: a = two int16 weights, b = bytes [lo0, lo1, hi0, hi1] of two int16 activations.
// lo: unsigned low bytes, hi: signed high bytes. sum w*x = 256 * sum w*hi + sum w*lo; both partial sums fit int32 because
// 255 * sum|w| <= 2^31 - 1 per output channel (checked at load, same bound as the CPU AVX-VNNI path).
__device__ __forceinline__ int dlo(unsigned a, unsigned b, int c) { int d; asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ int dhi(unsigned a, unsigned b, int c) { int d; asm("dp2a.hi.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ unsigned pk2(int x0, int x1) { return (unsigned) (x0 & 255) | ((unsigned) (x1 & 255) << 8) | ((unsigned) ((x0 >> 8) & 255) << 16) | ((unsigned) ((x1 >> 8) & 255) << 24); }

// yvvyee: activations travel between L7 kernels already in the conv input format: per channel pair cp, the reflect-padded
// (Cy+2) x (Cx+2) grid of words [lo0, lo1, hi0, hi1] (pk2), planes of PS words. (Cy+2)(Cx+2) = (th+14)(tw+14) is a multiple of 4
// because th, tw are multiples of 4, so a tile's NP*PS words copy as uint4.
template <int NTB>
__device__ void copyPK(const unsigned *__restrict__ src, int nwords, unsigned *PK)
{
  const uint4 *s = (const uint4 *) src; uint4 *d = (uint4 *) PK;
  for (int i = threadIdx.x; i < nwords / 4; i += NTB) d[i] = s[i];
}
// yvvyee: packed, reflect-padded output of one channel pair (planes a, b of Cy x Cx) into dst[PS]
template <int NTB>
__device__ void packOut(const short *a, const short *b, int Cy, int Cx, unsigned *__restrict__ dst)
{
  const int Wp = Cx + 2, PS = (Cy + 2) * Wp;
  for (int i = threadIdx.x; i < PS; i += NTB) { const int py = i / Wp, px = i - py * Wp, s = refl(py - 1, Cy) * Cx + refl(px - 1, Cx); dst[i] = pk2(a[s], b ? b[s] : 0); }
}
// yvvyee: packed weights for outputs o0..o0+OB-1 of layer l: WK[(t * NP + cp) * OB + j] = w[o][2cp][t] | w[o][2cp+1][t] << 16
template <int OB, int NTB>
__device__ void loadWK(const short *__restrict__ gW3, int l, int o0, unsigned *WK)
{
  const int Ci = P.c3Ci[l], NP = (Ci + 1) / 2;
  for (int i = threadIdx.x; i < 9 * NP * OB; i += NTB)
  {
    const int j = i % OB, cp = (i / OB) % NP, t = i / (OB * NP); const short *w = gW3 + P.c3off[l] + (o0 + j) * Ci * 9;
    const int w0 = w[2 * cp * 9 + t], w1 = 2 * cp + 1 < Ci ? w[(2 * cp + 1) * 9 + t] : 0;
    WK[i] = (unsigned) (w0 & 0xffff) | ((unsigned) w1 << 16);
  }
}
// yvvyee: 3x3 conv on the packed input for OB outputs. out = clamp16(rshr(bo + 256 * sum w*x, sh)) as in the reference (inputs are
// numerators x*256, up = 0). One thread = one pixel; weights are warp-uniform (broadcast loads).
template <int NP, int OB, int NTB>
__device__ void conv3pk(const unsigned *PK, int Cy, int Cx, int l, int o0, const unsigned *WK, int y0, int x0, int ny, int nx, short *o16, int *o32)
{
  const int npix = ny * nx, Wp = Cx + 2, PS = (Cy + 2) * Wp, CHW = Cy * Cx, sh = P.c3sh[l];
  for (int q = threadIdx.x; q < npix; q += NTB)
  {
    const int y = y0 + q / nx, x = x0 + q % nx; int sl[OB], s8[OB];
#pragma unroll
    for (int j = 0; j < OB; j++) { sl[j] = 0; s8[j] = 0; }
    const unsigned *base = PK + y * Wp + x;
#pragma unroll
    for (int t = 0; t < 9; t++)
    {
      const unsigned *bt = base + (t / 3) * Wp + (t % 3);
#pragma unroll
      for (int cp = 0; cp < NP; cp++)
      {
        const unsigned a = bt[cp * PS]; const unsigned *w = WK + (t * NP + cp) * OB;
#pragma unroll
        for (int j = 0; j < OB; j++) { const unsigned wj = w[j]; sl[j] = dlo(wj, a, sl[j]); s8[j] = dhi(wj, a, s8[j]); }
      }
    }
#pragma unroll
    for (int j = 0; j < OB; j++)
    {
      const long long S = (long long) s8[j] * 256 + sl[j]; const long long v = clamp16(rshrL(P.c3bo[l][o0 + j] + S * 256, sh));
      if (o16) o16[j * CHW + y * Cx + x] = (short) v; else o32[(y - y0) * nx + (x - x0)] = (int) v;
    }
  }
  __syncthreads();
}

// yvvyee: FiltAct for channels [c0, c0+nc) of ACT, in place. Zero-stuffed 2x upsample, hu (x2) h/v, lrelu(256/51), hd h/v, [::2].
// Only even source rows/cols of the upsampled grid are non-zero; reflection keeps parity, so output position xx uses taps k == xx (mod 2)
// (stages A, B). Only even output columns/rows are kept (D, E). int32 sums in A, B, E are exact: |input| <= 32767 and
// 32767*sum|taps| + 2^(s-1) < 2^31 (checked at load). Stage D accumulates the reference numerators (a*256 or a*51) times hd in int64,
// which is the reference expression itself. Mapping: one warp per row, lane columns {lane, lane+32}. Stage B (vertical) and stage D
// (horizontal on the same row) are fused per warp through a warp-private row buffer: 3 block barriers per group.
template <int NTB>
__device__ void filtActG(short *act, int c0, int nc, int Cy, int Cx, int l, short *TA, short *T2, int *WB)
{
  const int H2 = 2 * Cy, W2 = 2 * Cx, CHW = Cy * Cx, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, NW = NTB / 32;
  const int sA = P.faA[l], sB = P.faB[l], sD = P.faD[l], sE = P.faE[l];
  const int hA = 1 << (sA - 1), hB = 1 << (sB - 1), hE = 1 << (sE - 1); const long long hD = 1LL << (sD - 1);
  const int p = lane & 1, x1 = lane + 32; const bool has0 = lane < W2, has1 = x1 < W2, hasD = lane < Cx;
  int tpA[6], iA0[6], iA1[6], hd[12], iD[12], huE[6], huO[6];
#pragma unroll
  for (int j = 0; j < 6; j++) { tpA[j] = P.hu2[2 * j + p]; iA0[j] = refl(lane + p + 2 * j - 6, W2) >> 1; iA1[j] = refl(x1 + p + 2 * j - 6, W2) >> 1; huE[j] = P.hu2[2 * j]; huO[j] = P.hu2[2 * j + 1]; }
#pragma unroll
  for (int k = 0; k < 12; k++) { hd[k] = P.hd[k]; iD[k] = refl(2 * lane + k - 6, W2); }
  const int x1c = has1 ? x1 : lane;   // yvvyee: clamp the second column inside the row (its result is discarded)
  for (int c = 0, i = warp; c < nc; )   // A: TA[c][i][xx] = t[2i][xx]
  {
    const short *xr = act + (c0 + c) * CHW + i * Cx; short *d = TA + (c * Cy + i) * W2; int a0 = 0, a1 = 0;
#pragma unroll
    for (int j = 0; j < 6; j++) { a0 += xr[iA0[j]] * tpA[j]; a1 += xr[iA1[j]] * tpA[j]; }
    if (has0) d[lane] = (short) clamp16i((a0 + hA) >> sA);
    if (has1) d[x1] = (short) clamp16i((a1 + hA) >> sA);
    i += NW; while (i >= Cy) { i -= Cy; c++; }
  }
  __syncthreads();
  int *wb = WB + warp * 64;
  for (int c = 0, yy = warp; c < nc; )   // B: a[yy][xx] -> lrelu numerator in the warp buffer, then C/D: T2[c][yy][j] = t[yy][2j]
  {
    const int q = yy & 1, b0 = yy + q - 6; const short *ta = TA + c * Cy * W2; int a0 = 0, a1 = 0;
    if (b0 >= 0 && b0 + 10 <= H2 - 1)
    {
      const short *row = ta + (b0 >> 1) * W2;
#pragma unroll
      for (int j = 0; j < 6; j++) { const int t = q ? huO[j] : huE[j]; a0 += row[j * W2 + lane] * t; a1 += row[j * W2 + x1c] * t; }
    }
    else
    {
#pragma unroll
      for (int j = 0; j < 6; j++) { const short *row = ta + (refl(b0 + 2 * j, H2) >> 1) * W2; const int t = q ? huO[j] : huE[j]; a0 += row[lane] * t; a1 += row[x1c] * t; }
    }
    const int v0 = clamp16i((a0 + hB) >> sB), v1 = clamp16i((a1 + hB) >> sB);
    wb[lane] = v0 >= 0 ? v0 * 256 : v0 * 51; wb[x1] = v1 >= 0 ? v1 * 256 : v1 * 51;
    __syncwarp();
    long long acc = 0;
#pragma unroll
    for (int k = 0; k < 12; k++) acc += (long long) wb[iD[k]] * hd[k];
    if (hasD) T2[(c * H2 + yy) * Cx + lane] = (short) clamp16((acc + hD) >> sD);   // = rshr(sum num*hd, sD)
    __syncwarp();
    yy += NW; while (yy >= H2) { yy -= H2; c++; }
  }
  __syncthreads();
  for (int c = 0, i = warp; c < nc; )   // E: y[i][j] = a[2i][2j]
  {
    const short *t2 = T2 + c * H2 * Cx + lane; int acc = 0; const int b0 = 2 * i - 6;
    if (b0 >= 0 && b0 + 11 <= H2 - 1)
    {
      const short *col = t2 + b0 * Cx;
#pragma unroll
      for (int k = 0; k < 12; k++) acc += col[k * Cx] * hd[k];
    }
    else
    {
#pragma unroll
      for (int k = 0; k < 12; k++) acc += t2[refl(b0 + k, H2) * Cx] * hd[k];
    }
    if (hasD) act[(c0 + c) * CHW + i * Cx + lane] = (short) clamp16i((acc + hE) >> sE);
    i += NW; while (i >= Cy) { i -= Cy; c++; }
  }
  __syncthreads();
}

__global__ void __launch_bounds__(NT, 1) kFront(const NNInterp::BlockIn *__restrict__ gin, NNInterp::BlockOut *__restrict__ gout, int n, unsigned *__restrict__ gpk,
                                                const short *__restrict__ gW3, const int *__restrict__ lut, int outMask, int *err)
{
  extern __shared__ __align__(16) unsigned char sm[];
  const int tid = threadIdx.x, tile = blockIdx.x; if (tile >= n) return;
  const NNInterp::BlockIn &in = gin[tile]; NNInterp::BlockOut &out = gout[tile];
  const int tw = in.tw, th = in.th, stride = in.stride, Hh = th + 16, Ww = tw + 16, HW = Hh * Ww, Ly = th + 4, Lx = tw + 4, Cy = th + 12, Cx = tw + 12, NL = Ly * Lx, CHW = Cy * Cx;
  if (tw < 4 || th < 4 || tw > 16 || th > 16 || (tw & 3) || (th & 3) || stride < Ww || stride * (Hh - 1) + Ww > 1024) { if (tid == 0) atomicExch(err, 1); return; }
  const int st = in.st, qp = in.qp, fx = max(0, in.fx), fy = max(0, in.fy), imv = in.imv, merge = in.merge, dpoc = in.dpoc, w4 = in.w4, h4 = in.h4;

  long long *SD = (long long *) (sm + OFF_SMALL), *QTAB = SD + 16, *BASEA = QTAB + 64, *BASEQ = BASEA + 32;
  long long *XD = (long long *) (sm + OFF_XD); short *LPN = (short *) (sm + OFF_LPN), *CF = (short *) (sm + OFF_CF);
  unsigned char *S = sm + OFF_S;

  PROF(0);
  // ---- input window (Q10.4) and code map; block flag: |x| >= 2^31 needs the __int128 FIR
  long long *XB3 = (long long *) (S + A_XB3); int *CM = (int *) (S + A_PHI);
  int big = 0;
  for (int p = tid; p < HW; p += NT)
  {
    const int yy = p / Ww, xx = p - yy * Ww; const int v = in.win[yy * stride + xx];
    XB3[p] = (long long) v << PIXF; CM[p] = in.map[yy * stride + xx]; big |= (v >= (1 << 26) || v < -(1 << 26));
  }
  if (tid == 0)
  {
    for (int c = 0; c < 16; c++) SD[c] = 0;
    SD[0] = fdivRoundL((long long) qp * 16384, 63); SD[1 + clampi(imv, 0, 3)] = 16384;
    const bool integ = (fx == 0 && fy == 0), half = !integ && (fx % 8 == 0) && (fy % 8 == 0), quar = !integ && !half && (fx % 4 == 0) && (fy % 4 == 0);
    SD[5 + (integ ? 0 : half ? 1 : quar ? 2 : 3)] = 16384;
    SD[9] = merge ? 16384 : 0; SD[10] = clamp16(fdivRoundL((long long) abs(dpoc) * 16384, 16)); SD[11] = P.tabB[clampi(w4, 1, 32) - 1]; SD[12] = P.tabB[clampi(h4, 1, 32) - 1]; SD[13 + clampi(st, 0, 2)] = 16384;
  }
  if (tid >= 64 && tid < 128) QTAB[tid - 64] = clamp16(fdivRoundL((long long) (tid - 64 - qp) * 16384, 12));
  const bool xbig = __syncthreads_or(big);

  // ---- code map features (codemapFeats): rows by threads [0, Hh), columns by threads [128, 128 + Ww)
  {
    const bool horiz = tid < 128; const int a = horiz ? tid : tid - 128, n = horiz ? Ww : Hh, na = horiz ? Hh : Ww;
    if (a < na)
    {
      short *cfD = CF + (horiz ? 0 : 2) * 1024, *cfDb = CF + (horiz ? 1 : 3) * 1024, *cfV = CF + (horiz ? 4 : 6) * 1024, *cfVb = CF + (horiz ? 5 : 7) * 1024;
      int last = -1; int vb = 0;
      for (int i = 0; i < n; i++)
      {
        const int p = horiz ? a * Ww + i : i * Ww + a, c = CM[p]; const int flag = horiz ? (c & 1) : ((c >> 1) & 1), bs = horiz ? ((c >> 2) & 3) : ((c >> 4) & 3);
        if (flag) { last = i; vb = bs; }
        const int d = last >= 0 ? i - last : n; cfD[p] = (short) (min(d, 8) * 2048); cfV[p] = (short) (last >= 0 ? vb * 8192 : 0);
      }
      int nxt = n + 1, vf = 0;
      for (int i = n - 1; i >= 0; i--)
      {
        const int p = horiz ? a * Ww + i : i * Ww + a, c = CM[p];
        const int d = (nxt <= n) ? (nxt - 1 - i) : n; cfDb[p] = (short) (min(d, 8) * 2048); cfVb[p] = (short) ((nxt <= n) ? vf * 8192 : 0);
        const int flag = horiz ? (c & 1) : ((c >> 1) & 1), bs = horiz ? ((c >> 2) & 3) : ((c >> 4) & 3);
        if (flag) { nxt = i; vf = bs; }
      }
    }
  }
  for (int p = tid; p < HW; p += NT)
  {
    const int c = CM[p], valid = (c >> 14) & 1, qpr = (c >> 8) & 63;
    CF[8 * 1024 + p] = (short) (((c >> 6) & 1) * 16384); CF[9 * 1024 + p] = (short) (((c >> 7) & 1) * 16384);
    CF[10 * 1024 + p] = (short) (valid ? QTAB[qpr] : 0); CF[11 * 1024 + p] = (short) (valid * 16384);
  }
  // yvvyee: block-constant part of the two 1x1 heads (sd inputs), as in conv1x1x2F
  if (tid < 32)
  {
    long long a = P.n0bo[tid < P.n0.Co ? tid : 0], q = P.q0bo[tid < P.q0.Co ? tid : 0];
    for (int c = 0; c < 16; c++) { a += (long long) P.n0w[(tid < P.n0.Co ? tid : 0) * 28 + c] * SD[c]; q += ((long long) P.q0w[(tid < P.q0.Co ? tid : 0) * 24 + c] * SD[c]) << P.q0up[c]; }
    BASEA[tid] = a; BASEQ[tid] = q;
  }
  __syncthreads();

  PROF(1);
  // ---- L1 band split
  short *G3 = (short *) (S + A_G3); int *TMP = (int *) (S + A_TMP), *SQ = (int *) (S + A_SQ), *PHI = (int *) (S + A_PHI);
  for (int g = 0; g < 3; g++)
  {
    if (xbig) sepFir<true>(XB3, Hh, Ww, g, TMP, G3 + g * 1024, FIRF, FIRF, 32767);
    else sepFir<false>(XB3, Hh, Ww, g, TMP, G3 + g * 1024, FIRF, FIRF, 32767);
  }
  for (int p = tid; p < HW; p += NT) XB3[p] -= G3[p];   // B3 = x - g0
  __syncthreads();
  PROF(2);
  // ---- L2 band energy phi_b
  for (int b = 0; b < 4; b++)
  {
    for (int p = tid; p < HW; p += NT)
    {
      const long long B = b == 0 ? G3[2048 + p] : b == 1 ? (long long) G3[1024 + p] - G3[2048 + p] : b == 2 ? (long long) G3[p] - G3[1024 + p] : XB3[p];
      SQ[p] = (int) energyQ(B, B, P.eSq[b], 0);
    }
    __syncthreads();
    sepFir<false>(SQ, Hh, Ww, 3, TMP, PHI + b * 1024, FIRF + P.eSq[b] - P.ePhi[b], FIRF, 2147483647LL);
  }
  PROF(3);
  // ---- L3 noise query (1x1 relu -> 1x1) and L4 Wiener, per pixel
  for (int p = tid; p < HW; p += NT)
  {
    int cfv[12];
#pragma unroll
    for (int k = 0; k < 12; k++) cfv[k] = CF[k * 1024 + p];
    long long a2[8];
#pragma unroll
    for (int b = 0; b < 8; b++) a2[b] = P.n2bo[b];
    for (int o = 0; o < P.n0.Co; o++)
    {
      long long acc = BASEA[o]; const int *w = P.n0w + o * 28 + 16;
#pragma unroll
      for (int k = 0; k < 12; k++) acc += (long long) w[k] * cfv[k];
      long long h = clamp16(rshrL(acc, P.n0sh)); if (h < 0) h = 0;
#pragma unroll
      for (int b = 0; b < 8; b++) if (b < P.n2.Co) a2[b] += (long long) P.n2w[b * 32 + o] * h;
    }
    i128 acc = 0;
#pragma unroll
    for (int b = 0; b < 4; b++)
    {
      const long long h2 = clamp16(rshrL(a2[b], P.n2sh)), s2 = P.n2.oexp;
      const long long lphi = clamp16(rshrW(((i128) h2 << 8) - ((i128) 7680 << s2), (int) s2));
      long long t; int nn; expQ(lut, lphi, t, nn); const long long phin = clamp32(rshrL(t, 15 - nn - P.ePhin));
      const int Kn = max(P.ePhin, 40); const i128 num = ((i128) phin << (Kn - P.ePhin)) + ((i128) 1 << (Kn - 40));
      LPN[b * 1024 + p] = (short) clamp16(rshrW((i128) lnQ10(lut, num, Kn) * P.INV30, 12));
      const long long phi = PHI[b * 1024 + p]; const int Kc = max(max(P.ePhi[b], P.ePhin), 40);
      const i128 den = ((i128) phi << (Kc - P.ePhi[b])) + ((i128) phin << (Kc - P.ePhin)) + ((i128) 1 << (Kc - 40));
      long long r15; int e; recipQ(lut, den, Kc, r15, e);
      const long long G15 = clamp32(rshrW((i128) phi * r15, P.ePhi[b] + e));
      const long long B = b == 0 ? G3[2048 + p] : b == 1 ? (long long) G3[1024 + p] - G3[2048 + p] : b == 2 ? (long long) G3[p] - G3[1024 + p] : XB3[p];
      acc += (i128) G15 * B;
    }
    XD[p] = rshrW(acc, 15);
  }
  __syncthreads();
  PROF(4);
  // ---- L4b half-pel low-pass (AMVR 3, per axis); temp in XB3 (dead)
  {
    const bool ax = (imv == 3 && fx == 8), ay = (imv == 3 && fy == 8);
    for (int axis = 0; axis < 2; axis++)
    {
      if (!(axis == 0 ? ax : ay)) continue;
      for (int p = tid; p < HW; p += NT)
      {
        const int yy = p / Ww, xx = p - yy * Ww; i128 acc = 0;
        for (int k = 0; k < 7; k++) acc += (i128) (axis == 0 ? XD[yy * Ww + clampi(xx + k - 3, 0, Ww - 1)] : XD[clampi(yy + k - 3, 0, Hh - 1) * Ww + xx]) * P.k7[k];
        XB3[p] = rshrW(acc, FIRF);
      }
      __syncthreads();
      for (int p = tid; p < HW; p += NT) XD[p] = rshrW(((i128) XD[p] << 15) + (i128) P.gate * (XB3[p] - XD[p]), 15);
      __syncthreads();
    }
  }
  if (outMask & 4) for (int p = tid; p < HW; p += NT) out.xd[p] = XD[p];

  PROF(5);
  // ---- L6 query input: structure tensor -> gauss10 -> log1p (only the (th+4)x(tw+4) attention pixels)
  int *S0 = (int *) (S + B_S0), *TMP2 = (int *) (S + B_TMP), *S1V = (int *) (S + B_S1V); short *LST = (short *) (S + B_LST), *Q = (short *) (S + B_Q);
  for (int c = 0; c < 3; c++)
  {
    for (int p = tid; p < HW; p += NT)
    {
      const int yy = p / Ww, xx = p - yy * Ww;
      const long long gx = XD[yy * Ww + min(xx + 1, Ww - 1)] - XD[yy * Ww + max(xx - 1, 0)], gy = XD[min(yy + 1, Hh - 1) * Ww + xx] - XD[max(yy - 1, 0) * Ww + xx];
      S0[p] = (int) energyQ(c == 2 ? gy : gx, c == 0 ? gx : gy, P.eSt0, 2);
    }
    __syncthreads();
    sepFir<false>(S0, Hh, Ww, 3, TMP2, S1V, FIRF + P.eSt0 - P.eSt, FIRF, 2147483647LL);
    for (int t = tid; t < NL; t += NT)
    {
      const int p = (6 + t / Lx) * Ww + 6 + t % Lx; const long long v = (long long) S1V[p] * 1023; const int sg = v < 0 ? -1 : (v > 0 ? 1 : 0);
      const i128 num = (i128) (v < 0 ? -v : v) + ((i128) 1 << P.eSt);
      LST[c * 400 + t] = (short) clamp16(rshrW((i128) sg * lnQ10(lut, num, P.eSt), 2));
    }
    __syncthreads();
  }
  PROF(6);
  // ---- L6 query 1x1 head: [sd(16), lst(3), lpn3, cf0..3] -> relu -> q (only attention pixels)
  for (int t = tid; t < NL; t += NT)
  {
    const int p = (6 + t / Lx) * Ww + 6 + t % Lx; int xv[8];
    xv[0] = LST[t]; xv[1] = LST[400 + t]; xv[2] = LST[800 + t]; xv[3] = LPN[3 * 1024 + p];
#pragma unroll
    for (int k = 0; k < 4; k++) xv[4 + k] = CF[k * 1024 + p];
    long long a2[8];
#pragma unroll
    for (int j = 0; j < 8; j++) a2[j] = P.q2bo[j];
    for (int o = 0; o < P.q0.Co; o++)
    {
      long long acc = BASEQ[o]; const int *w = P.q0w + o * 24 + 16;
#pragma unroll
      for (int k = 0; k < 8; k++) acc += ((long long) w[k] * xv[k]) << P.q0up[16 + k];
      long long h = clamp16(rshrL(acc, P.q0sh)); if (h < 0) h = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) if (j < P.q2.Co) a2[j] += (long long) P.q2w[j * 32 + o] * h;
    }
#pragma unroll
    for (int j = 0; j < 5; j++) Q[j * 400 + t] = (short) clamp16(rshrL(a2[j], P.q2sh));
  }
  PROF(7);
  // ---- L5 horizontal passes: one per distinct horizontal shift, columns 2..2+Cx (shared with the L7 aligned window)
  const bool xdbig = __syncthreads_or([&] { int b = 0; for (int p = tid; p < HW; p += NT) b |= (XD[p] >= (1LL << 40) || XD[p] <= -(1LL << 40)); return b; }());
  long long *XH = (long long *) (S + B_XH); const int nh = P.nh, XHS = Hh * Cx;
  int hbig = 0;
  for (int it = tid; it < nh * XHS; it += NT)
  {
    const int h = it / XHS, r = it - h * XHS, yy = r / Cx, t = r - yy * Cx;
    const int p16 = (2 + t) * 16 + fx + P.hsx[h], i0 = p16 >> 4, f = p16 & 15; const long long *xr = XD + yy * Ww; const int *kv = P.k12 + f * 12; long long v;
    if (xdbig) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) xr[clampi(i0 + k - 5, 0, Ww - 1)] * kv[k]; v = rshrW(acc, FIRF); }
    else { long long acc = 0;
#pragma unroll
      for (int k = 0; k < 12; k++) acc += xr[clampi(i0 + k - 5, 0, Ww - 1)] * kv[k];
      v = rshrL(acc, FIRF); }
    XH[it] = v; hbig |= (v >= (1LL << 40) || v <= -(1LL << 40));
  }
  const bool vwide = __syncthreads_or(hbig) || xdbig;
  PROF(8);
  // ---- L5 vertical passes (on the fly) + L6 attention
  for (int t = tid; t < NL; t += NT)
  {
    const int ty = t / Lx, tx = t - ty * Lx; long long qv[5];
#pragma unroll
    for (int j = 0; j < 5; j++) qv[j] = Q[j * 400 + t];
    long long lg[KMAX], mx = LLONG_MIN;
#pragma unroll
    for (int c = 0; c < KMAX; c++)
    {
      if (c < P.K) {
      long long a = 0;
#pragma unroll
      for (int j = 0; j < 5; j++) a += qv[j] * P.phiM[c * 5 + j];
      lg[c] = rshrW((i128) a, P.ephi + P.q2.oexp - 8); mx = lg[c] > mx ? lg[c] : mx; }
    }
    long long e45[KMAX]; i128 sum = 0;
#pragma unroll
    for (int c = 0; c < KMAX; c++)
    {
      if (c < P.K) { long long tt; int nn; expQ(lut, lg[c] - mx, tt, nn); e45[c] = rshrL(tt, 15 - nn - 45); sum += e45[c]; }
    }
    long long r15; int e; recipQ(lut, sum, 45, r15, e); i128 acc = 0;
#pragma unroll
    for (int c = 0; c < KMAX; c++)
    {
      if (c >= P.K) continue;
      const long long a15 = clamp32(rshrW((i128) e45[c] * r15, 45 + e));
      const int p16 = (6 + ty) * 16 + fy + P.cdy[c], i0 = p16 >> 4, f = p16 & 15; const long long *xh = XH + P.hid[c] * XHS + 4 + tx; const int *kv = P.k12 + f * 12; long long vc;
      if (vwide) { i128 s = 0; for (int k = 0; k < 12; k++) s += (i128) xh[clampi(i0 + k - 5, 0, Hh - 1) * Cx] * kv[k]; vc = rshrW(s, FIRF); }
      else { long long s = 0;
#pragma unroll
        for (int k = 0; k < 12; k++) s += xh[clampi(i0 + k - 5, 0, Hh - 1) * Cx] * kv[k];
        vc = rshrL(s, FIRF); }
      acc += (i128) a15 * vc;
    }
    out.ylin[t] = rshrW(acc, 15);   // yvvyee: always stored (read by kConv4L8)
  }
  PROF(9);
  // ---- L7 input (act0): ch0 = aligned window (shift fx, fy) rows/cols 2..2+C as Q14 of the normalized value, ch1..4 = lpn,
  // ch5..10 = cf{0,1,2,3,8,9} (all cropped to rows/cols 2..2+C), ch11 = 0; written packed and reflect-padded (6 channel pairs)
  short *CIN0 = (short *) (S + B_LST);
  __syncthreads();
  for (int it = tid; it < CHW; it += NT)
  {
    const int ty = it / Cx, tx = it - ty * Cx; const int p16 = (2 + ty) * 16 + fy, i0 = p16 >> 4, f = p16 & 15; const long long *xh = XH + tx; const int *kv = P.k12 + f * 12; long long aw;
    if (vwide) { i128 s = 0; for (int k = 0; k < 12; k++) s += (i128) xh[clampi(i0 + k - 5, 0, Hh - 1) * Cx] * kv[k]; aw = rshrW(s, FIRF); }
    else { long long s = 0;
#pragma unroll
      for (int k = 0; k < 12; k++) s += xh[clampi(i0 + k - 5, 0, Hh - 1) * Cx] * kv[k];
      aw = rshrL(s, FIRF); }
    CIN0[it] = (short) clamp16(rdivPow2((i128) aw << P.eIn, 14));
  }
  __syncthreads();
  {
    unsigned *gp = gpk + (size_t) tile * PKW; const int Wp = Cx + 2, PS = (Cy + 2) * Wp; const int cfi[6] = { 0, 1, 2, 3, 8, 9 };
    for (int i = tid; i < 6 * PS; i += NT)
    {
      const int cp = i / PS, r = i - cp * PS, py = r / Wp, px = r - py * Wp, sy = refl(py - 1, Cy), sx = refl(px - 1, Cx), s = (sy + 2) * Ww + sx + 2; int v[2];
#pragma unroll
      for (int h = 0; h < 2; h++) { const int c = 2 * cp + h; v[h] = c == 0 ? CIN0[sy * Cx + sx] : c <= 4 ? LPN[(c - 1) * 1024 + s] : c <= 10 ? CF[cfi[c - 5] * 1024 + s] : 0; }
      gp[i] = pk2(v[0], v[1]);
    }
  }
  PROF(10);
}

__device__ __forceinline__ bool badTile(int tw, int th) { return tw < 4 || th < 4 || tw > 16 || th > 16 || (tw & 3) || (th & 3); }

// yvvyee: L7 layer l < 4: conv (outputs o0, o0+1, all Cy x Cx pixels) followed by FiltAct on those two channels, written as the packed
// channel pair og of the next layer's input. grid = (Co / 2, tiles): the blocks of one tile are adjacent in launch order, so its
// packed input is read from DRAM once and then hits L2 (tile-major order re-read it from DRAM per block)
template <int NP>
__global__ void __launch_bounds__(NTC) kConvFA(int l, const unsigned *__restrict__ pkIn, unsigned *__restrict__ pkOut, const NNInterp::BlockIn *__restrict__ gin, const short *__restrict__ gW3, int n)
{
  extern __shared__ __align__(16) unsigned char sm[];
  const int tile = blockIdx.y, og = blockIdx.x, o0 = og * OPB; if (tile >= n) return;
  const int tw = gin[tile].tw, th = gin[tile].th; if (badTile(tw, th)) return;
  const int Cy = th + 12, Cx = tw + 12, CHW = Cy * Cx, PS = (Cy + 2) * (Cx + 2);
  unsigned *PK = (unsigned *) (sm + L_PK), *WK = (unsigned *) (sm + L_WK); short *PL = (short *) (sm + L_PL);
  short *TA = (short *) (sm + L_PK), *T2 = TA + OPB * 28 * 56; int *WB = (int *) (T2 + OPB * 56 * 28);
  copyPK<NTC>(pkIn + (size_t) tile * PKW, NP * PS, PK);
  loadWK<OPB, NTC>(gW3, l, o0, WK);
  __syncthreads();
  conv3pk<NP, OPB, NTC>(PK, Cy, Cx, l, o0, WK, 0, 0, Cy, Cx, PL, nullptr);
  filtActG<NTC>(PL, 0, OPB, Cy, Cx, l, TA, T2, WB);
  packOut<NTC>(PL, PL + CHW, Cy, Cx, pkOut + (size_t) tile * PKW + og * PS);
}

// yvvyee: last conv (one output, tile region only) and L8 rate gate per 4x4 block: c = D r D^T, sigmoid soft threshold, r' = D^T c' D.
// y goes to ycomp[tile*256] when ycomp is given (compact copy-back), else to gout[tile].y
__global__ void __launch_bounds__(NTC) kConv4L8(const unsigned *__restrict__ pkIn, const NNInterp::BlockIn *__restrict__ gin, NNInterp::BlockOut *__restrict__ gout,
                                                const short *__restrict__ gW3, const int *__restrict__ lut, int n, long long *__restrict__ ycomp)
{
  extern __shared__ __align__(16) unsigned char sm[];
  const int tile = blockIdx.x, tid = threadIdx.x; if (tile >= n) return;
  const int tw = gin[tile].tw, th = gin[tile].th, qp = gin[tile].qp; if (badTile(tw, th)) return;
  const int Cy = th + 12, Cx = tw + 12, Lx = tw + 4, PS = (Cy + 2) * (Cx + 2);
  unsigned *PK = (unsigned *) (sm + L_PK), *WK = (unsigned *) (sm + L_WK); int *C4 = (int *) (sm + L_C4); long long *CC = (long long *) (sm + L_CC), *CC2 = CC + 256;
  copyPK<NTC>(pkIn + (size_t) tile * PKW, 8 * PS, PK);
  loadWK<1, NTC>(gW3, 4, 0, WK);
  __syncthreads();
  conv3pk<8, 1, NTC>(PK, Cy, Cx, 4, 0, WK, 6, 6, th, tw, nullptr, C4);
  NNInterp::BlockOut &out = gout[tile]; long long *yd = ycomp ? ycomp + (size_t) tile * 256 : (long long *) out.y;
  const int nbx = tw / 4, nblk = nbx * (th / 4), eD = P.eD, qc = clampi(qp, 0, 63), eth = P.eth[qc]; const long long thq = P.rth[qc], r15th = P.r15th[qc];
  const int kdd = 29 + (eth >= 0 ? eth : 0);
  for (int it = tid; it < nblk * 16; it += NTC)
  {
    const int blk = it >> 4, i = (it >> 2) & 3, j = it & 3, by = (blk / nbx) * 4, bx = (blk % nbx) * 4; i128 acc = 0;
    for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) acc += (i128) P.D[i * 4 + k] * C4[(by + k) * tw + bx + l] * P.D[j * 4 + l];
    const long long c = rshrW(acc * (1024LL << PIXF), 2 * eD + P.sc4); CC[it] = c;
    const long long ac = c < 0 ? -c : c;
    i128 nn = (i128) 4 * (ac - thq) * r15th * 256; if (eth < 0) nn <<= (-eth);
    const long long arg = clamp32(rdivPow2(nn, kdd));
    CC2[it] = rshrW((i128) c * sigQ15(lut, arg), 15);
  }
  __syncthreads();
  for (int it = tid; it < nblk * 16; it += NTC)
  {
    const int blk = it >> 4, i = (it >> 2) & 3, j = it & 3, by = (blk / nbx) * 4, bx = (blk % nbx) * 4; i128 acc = 0;
    for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) acc += (i128) P.D[k * 4 + i] * CC2[blk * 16 + k * 4 + l] * P.D[l * 4 + j];
    yd[(by + i) * tw + bx + j] = out.ylin[(by + i + 2) * Lx + bx + j + 2] + rshrW(acc, 2 * eD);
  }
}

// ---------------------------------------------------------------- host side
struct Arr { std::vector<long long> v; std::vector<long long> shape; };
bool loadManifest(const std::string &dir, std::map<std::string, Arr> &m)
{
  FILE *mf = fopen((dir + "/manifest.txt").c_str(), "r"); if (!mf) return false;
  char name[256], dtype[32], fn[256]; int nd;
  while (fscanf(mf, "%255s %31s %d", name, dtype, &nd) == 3)
  {
    Arr a; long long cnt = 1;
    for (int i = 0; i < nd; i++) { long long d; if (fscanf(mf, "%lld", &d) != 1) { fclose(mf); return false; } a.shape.push_back(d); cnt *= d; }
    if (fscanf(mf, "%255s", fn) != 1) { fclose(mf); return false; }
    FILE *f = fopen((dir + "/" + fn).c_str(), "rb"); if (!f) { fclose(mf); return false; } a.v.resize(cnt);
    bool ok;
    if (std::string(dtype) == "int16") { std::vector<int16_t> t(cnt); ok = fread(t.data(), 2, cnt, f) == (size_t) cnt; for (long long i = 0; i < cnt; i++) a.v[i] = t[i]; }
    else { std::vector<int32_t> t(cnt); ok = fread(t.data(), 4, cnt, f) == (size_t) cnt; for (long long i = 0; i < cnt; i++) a.v[i] = t[i]; }
    fclose(f); if (!ok) { fclose(mf); return false; } m[name] = a;
  }
  fclose(mf); return !m.empty();
}
long long hFdivRound(i128 num, i128 den) { i128 n2 = 2 * num + den, d2 = 2 * den; return (long long) ((n2 >= 0) ? n2 / d2 : -((-n2 + d2 - 1) / d2)); }
} // namespace

struct NNInterpCuda::Impl
{
  int dev = 0; cudaStream_t s = nullptr;
  short *dW3 = nullptr; int *dLut = nullptr; int *dErr = nullptr; int *hErr = nullptr;   // yvvyee: error flag read back asynchronously into pinned memory
  NNInterp::BlockIn *dIn = nullptr; NNInterp::BlockOut *dOut = nullptr; unsigned *dPk[2] = { nullptr, nullptr }; long long *dY = nullptr, *hY = nullptr; int cap = 0;   // yvvyee: L7 ping-pong packed activations [cap][PKW], compact y (device, pinned host)
  size_t paramBytes = 0;
};

NNInterpCuda::~NNInterpCuda()
{
  if (!m) return;
  cudaSetDevice(m->dev);
  cudaFree(m->dW3); cudaFree(m->dLut); cudaFree(m->dErr); cudaFreeHost(m->hErr); cudaFree(m->dIn); cudaFree(m->dOut); cudaFree(m->dPk[0]); cudaFree(m->dPk[1]); cudaFree(m->dY); cudaFreeHost(m->hY); if (m->s) cudaStreamDestroy(m->s);
  delete m;
}
void *NNInterpCuda::allocPinned(size_t bytes) { void *p = nullptr; return cudaMallocHost(&p, bytes) == cudaSuccess ? p : nullptr; }
void NNInterpCuda::freePinned(void *p) { cudaFreeHost(p); }
size_t NNInterpCuda::deviceBytes() const { return m ? m->paramBytes + (size_t) m->cap * (sizeof(BlockIn) + sizeof(BlockOut) + 2 * PKW * sizeof(unsigned) + 256 * sizeof(long long)) : 0; }

bool NNInterpCuda::load(const std::string &dir, int device)
{
  m_loaded = false;
  std::map<std::string, Arr> t; if (!loadManifest(dir, t)) { snprintf(m_err, sizeof m_err, "manifest load failed: %s", dir.c_str()); return false; }
  auto has = [&](const std::string &n) { return t.count(n) > 0; };
  auto A = [&](const std::string &n) -> const Arr & { return t.at(n); };
  auto Sv = [&](const std::string &n, int i = 0) { return A(n).v.at(i); };
  auto fail = [&](const char *why) { snprintf(m_err, sizeof m_err, "%s", why); return false; };
  static const char *req[] = { "fir_gauss06", "fir_gauss12", "fir_gauss24", "fir_gauss10", "fir_kaiser16x12", "alt_kernel7", "alt_gate_q15", "cand_dx_q4", "cand_dy_q4", "mat_phi_i16", "mat_phi_exp", "mat_d4_i16", "mat_d4_exp",
    "tab_log2bsz_q14", "rate_th_q4_by_qp", "fir_filtact_hu_half", "fir_filtact_hd", "corr_in_exp", "energy_phin_exp", "energy_st0_exp", "energy_st_exp", "lut_exp2_frac_q15", "lut_log2_mant_q10", "lut_recip_mant_q15", "lut_sigmoid_q15",
    "const_log2e_q16", "const_ln2_q16", "const_inv30_q16" };
  for (const char *n : req) if (!has(n)) return fail("missing array");
  for (int b = 0; b < 4; b++) if (!has("energy_sq" + std::to_string(b) + "_exp") || !has("energy_phi" + std::to_string(b) + "_exp")) return fail("missing energy exponent");

  DevP *p = new DevP(); memset(p, 0, sizeof(DevP));
  // FIR taps; int64 FIR path needs sum|h| < 2^24 and T <= 32
  const char *firs[4] = { "fir_gauss06", "fir_gauss12", "fir_gauss24", "fir_gauss10" };
  for (int f = 0; f < 4; f++)
  {
    const Arr &h = A(firs[f]); if (h.v.size() > 32) { delete p; return fail("FIR too long"); }
    long long l1 = 0; p->firN[f] = (int) h.v.size(); for (size_t k = 0; k < h.v.size(); k++) { p->firT[f][k] = (int) h.v[k]; l1 += llabs(h.v[k]); }
    if (l1 >= (1LL << 24)) { delete p; return fail("FIR L1 bound"); }
  }
  const Arr &k12 = A("fir_kaiser16x12"); if (k12.v.size() != 192) { delete p; return fail("kaiser size"); }
  for (int i = 0; i < 192; i++) { if (k12.v[i] < -32768 || k12.v[i] > 32767) { delete p; return fail("kaiser range"); } p->k12[i] = (int) k12.v[i]; }
  const Arr &k7 = A("alt_kernel7"); if (k7.v.size() != 7) { delete p; return fail("k7 size"); }
  for (int i = 0; i < 7; i++) p->k7[i] = (int) k7.v[i];
  p->gate = Sv("alt_gate_q15");
  const Arr &cdx = A("cand_dx_q4"), &cdy = A("cand_dy_q4"), &phiM = A("mat_phi_i16");
  p->K = (int) cdx.shape[0]; if (p->K < 1 || p->K > KMAX || (int) cdy.v.size() < p->K || (int) phiM.v.size() < p->K * 5) { delete p; return fail("candidate count"); }
  p->nh = 0; p->hsx[p->nh++] = 0;   // yvvyee: index 0 = shift fx itself (L7 aligned window), as hget(fx) in the CPU fast path
  for (int c = 0; c < p->K; c++)
  {
    int h = -1; for (int i = 0; i < p->nh; i++) if (p->hsx[i] == (int) cdx.v[c]) h = i;
    if (h < 0) { if (p->nh >= NHMAX) { delete p; return fail("too many distinct horizontal shifts"); } h = p->nh; p->hsx[p->nh++] = (int) cdx.v[c]; }
    p->hid[c] = h; p->cdy[c] = (int) cdy.v[c];
    for (int j = 0; j < 5; j++) p->phiM[c * 5 + j] = (int) phiM.v[c * 5 + j];
  }
  for (int b = 0; b < 4; b++) { p->eSq[b] = (int) Sv("energy_sq" + std::to_string(b) + "_exp"); p->ePhi[b] = (int) Sv("energy_phi" + std::to_string(b) + "_exp"); }
  p->ePhin = (int) Sv("energy_phin_exp"); p->eSt0 = (int) Sv("energy_st0_exp"); p->eSt = (int) Sv("energy_st_exp"); p->ephi = (int) Sv("mat_phi_exp"); p->eD = (int) Sv("mat_d4_exp"); p->eIn = (int) Sv("corr_in_exp");
  p->LOG2E = Sv("const_log2e_q16"); p->LN2 = Sv("const_ln2_q16"); p->INV30 = Sv("const_inv30_q16");
  // yvvyee: int64 FIR shifts must lie in [1, 62] so that int64 rshr equals the reference __int128 rshr
  for (int b = 0; b < 4; b++) { const int s1 = FIRF + p->eSq[b] - p->ePhi[b]; if (s1 < 1 || s1 > 62) { delete p; return fail("L2 FIR shift"); } }
  { const int s1 = FIRF + p->eSt0 - p->eSt; if (s1 < 1 || s1 > 62) { delete p; return fail("L6 FIR shift"); } }
  const Arr &tabB = A("tab_log2bsz_q14"), &rth = A("rate_th_q4_by_qp"), &D = A("mat_d4_i16");
  if (tabB.v.size() < 32 || rth.v.size() < 64 || D.v.size() < 16) { delete p; return fail("table sizes"); }
  for (int i = 0; i < 32; i++) p->tabB[i] = tabB.v[i];
  for (int i = 0; i < 16; i++) p->D[i] = (int) D.v[i];
  const long long PIX_DEN = 1024LL << PIXF;
  const Arr &lrec = A("lut_recip_mant_q15");
  for (int q = 0; q < 64; q++)   // yvvyee: per-QP threshold exponent and reciprocal, same code as predictRef (depends on QP only)
  {
    const long long thq = rth.v[q]; p->rth[q] = thq;
    int eth = -60; while (((i128) PIX_DEN << (eth + 1 + 60)) <= ((i128) thq << 60)) eth++;
    i128 numm = (i128) thq * 1024, denm = (i128) PIX_DEN; if (eth >= 0) denm <<= eth; else numm <<= (-eth);
    const int m10 = (int) hFdivRound(numm, denm); if (m10 < 1024 || m10 > 2048) { delete p; return fail("rate threshold mantissa"); }
    p->eth[q] = eth; p->r15th[q] = lrec.v[m10 - 1024];
  }
  // 1x1 heads
  auto mk1 = [&](const std::string &nm, C1D &L, int *w, int wcap, long long *bo, int bcap) {
    if (!has(nm + "_w") || !has(nm + "_b_q14") || !has(nm + "_wexp") || !has(nm + "_oexp")) return false;
    const Arr &wa = A(nm + "_w"), &ba = A(nm + "_b_q14"); if (wa.shape.size() < 2) return false;
    L.Co = (int) wa.shape[0]; L.Ci = (int) wa.shape[1]; L.we = (int) Sv(nm + "_wexp"); L.oexp = (int) Sv(nm + "_oexp");
    if ((long long) wa.v.size() != (long long) L.Co * L.Ci || L.Co * L.Ci > wcap || L.Co > bcap || (int) ba.v.size() < L.Co) return false;
    for (int i = 0; i < L.Co * L.Ci; i++) { if (wa.v[i] < -32768 || wa.v[i] > 32767) return false; }
    (void) bo; (void) w; return true; };
  if (!mk1("conv_noise_0", p->n0, p->n0w, 32 * 28, p->n0bo, 32) || !mk1("conv_noise_2", p->n2, p->n2w, 8 * 32, p->n2bo, 8) || !mk1("conv_dq_0", p->q0, p->q0w, 32 * 24, p->q0bo, 32) || !mk1("conv_dq_2", p->q2, p->q2w, 8 * 32, p->q2bo, 8)) { delete p; return fail("1x1 layer"); }
  if (p->n0.Ci != 28 || p->n0.Co > 32 || p->n2.Ci != p->n0.Co || p->n2.Co < 4 || p->n2.Co > 8 || p->q0.Ci != 24 || p->q0.Co > 32 || p->q2.Ci != p->q0.Co || p->q2.Co < 5 || p->q2.Co > 8) { delete p; return fail("1x1 shapes"); }
  {
    auto bias = [](long long b, int s) { return s >= 0 ? (b << s) : (b >> (-s)); };
    const Arr &w0 = A("conv_noise_0_w"), &b0 = A("conv_noise_0_b_q14"), &w2 = A("conv_noise_2_w"), &b2 = A("conv_noise_2_b_q14");
    const Arr &v0 = A("conv_dq_0_w"), &c0 = A("conv_dq_0_b_q14"), &v2 = A("conv_dq_2_w"), &c2 = A("conv_dq_2_b_q14");
    // noise head: all inputs at grid 14 -> SM = 14, up = 0
    for (int o = 0; o < p->n0.Co; o++) { p->n0bo[o] = bias(b0.v[o], p->n0.we); for (int c = 0; c < 28; c++) p->n0w[o * 28 + c] = (int) w0.v[o * 28 + c]; }
    p->n0sh = p->n0.we + 14 - p->n0.oexp;
    const int s1 = p->n0.oexp;
    for (int o = 0; o < p->n2.Co; o++) { p->n2bo[o] = bias(b2.v[o], p->n2.we + s1 - 14); for (int c = 0; c < p->n0.Co; c++) p->n2w[o * 32 + c] = (int) w2.v[o * p->n0.Co + c]; }
    p->n2sh = p->n2.we + s1 - p->n2.oexp;
    // query head: inputs sd(16)@14, lst(3)@8, lpn3@14, cf0..3@14 -> SM = 14
    const int sIn[24] = { 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 8, 8, 8, 14, 14, 14, 14, 14 };
    for (int c = 0; c < 24; c++) p->q0up[c] = 14 - sIn[c];
    for (int o = 0; o < p->q0.Co; o++) { p->q0bo[o] = bias(c0.v[o], p->q0.we); for (int c = 0; c < 24; c++) p->q0w[o * 24 + c] = (int) v0.v[o * 24 + c]; }
    p->q0sh = p->q0.we + 14 - p->q0.oexp;
    const int sq1 = p->q0.oexp;
    for (int o = 0; o < p->q2.Co; o++) { p->q2bo[o] = bias(c2.v[o], p->q2.we + sq1 - 14); for (int c = 0; c < p->q0.Co; c++) p->q2w[o * 32 + c] = (int) v2.v[o * p->q0.Co + c]; }
    p->q2sh = p->q2.we + sq1 - p->q2.oexp;
  }
  // 3x3 layers and FiltAct
  const Arr &hu = A("fir_filtact_hu_half"), &hd = A("fir_filtact_hd"); if (hu.v.size() != 12 || hd.v.size() != 12) { delete p; return fail("filtact taps"); }
  long long L1u = 0, L1d = 0; for (int k = 0; k < 12; k++) { L1u += llabs(2 * hu.v[k]); L1d += llabs(hd.v[k]); p->hu2[k] = (int) (2 * hu.v[k]); p->hd[k] = (int) hd.v[k]; }
  if (p->eIn != 14) { delete p; return fail("corr_in_exp != 14 (per-channel input scale not supported)"); }
  const char *cn[5] = { "conv_corr_0", "conv_corr_2", "conv_corr_4", "conv_corr_6", "conv_corr_8" }, *fn[4] = { "filtact_corr_1", "filtact_corr_3", "filtact_corr_5", "filtact_corr_7" };
  std::vector<short> w3; int sIn = p->eIn;
  for (int l = 0; l < 5; l++)
  {
    const std::string nm = cn[l]; if (!has(nm + "_w") || !has(nm + "_b_q14") || !has(nm + "_wexp") || !has(nm + "_oexp")) { delete p; return fail("conv3 arrays"); }
    const Arr &w = A(nm + "_w"), &b = A(nm + "_b_q14"); if (w.shape.size() != 4 || w.shape[2] != 3 || w.shape[3] != 3) { delete p; return fail("conv3 shape"); }
    const int Co = (int) w.shape[0], Ci = (int) w.shape[1], we = (int) Sv(nm + "_wexp"), so = (int) Sv(nm + "_oexp");
    if (Co > 16 || Ci > 16 || (l == 0 && Ci != 11) || (l > 0 && Ci != p->c3Co[l - 1]) || (l == 4 && Co != 1) || (int) b.v.size() < Co) { delete p; return fail("conv3 channels (device kernel holds 16 planes)"); }
    p->c3Co[l] = Co; p->c3Ci[l] = Ci; p->c3off[l] = (int) w3.size();
    const int SM = sIn + 8; p->c3sh[l] = we + SM - so;
    for (int o = 0; o < Co; o++) p->c3bo[l][o] = (we + SM - 14 >= 0) ? (b.v[o] << (we + SM - 14)) : (b.v[o] >> (14 - we - SM));
    for (long long v : w.v) { if (v < -32768 || v > 32767) { delete p; return fail("conv3 weight range"); } w3.push_back((short) v); }
    for (int o = 0; o < Co; o++) { long long l1 = 0; for (int i = 0; i < Ci * 9; i++) l1 += llabs(w.v[(size_t) o * Ci * 9 + i]); if (255 * l1 > INT32_MAX) { delete p; return fail("conv3 dp2a int32 bound"); } }   // yvvyee: |sum w*lo| <= 255*L1, |sum w*hi| <= 128*L1
    if ((l < 4 && (Co != 16 || Co % OPB)) || (l == 0 && (Ci + 1) / 2 != 6) || (l > 0 && Ci != 16)) { delete p; return fail("L7 channel layout (kernels are instantiated for 11->16->16->16->16->1)"); }
    if (l == 4) p->sc4 = so;
    if (l < 4)
    {
      const std::string fnm = fn[l]; if (!has(fnm + "_oexp") || A(fnm + "_oexp").v.size() < 2) { delete p; return fail("filtact exponent"); }
      const int eh = (int) Sv(fnm + "_oexp", 0), ev = (int) Sv(fnm + "_oexp", 1);
      p->faA[l] = FIRF + so - eh; p->faB[l] = FIRF + eh - ev; p->faD[l] = FIRF + 8 + ev - eh; p->faE[l] = FIRF + eh - ev;
      if (p->faA[l] < 1 || p->faA[l] > 30 || p->faB[l] < 1 || p->faB[l] > 30 || p->faE[l] < 1 || p->faE[l] > 30 || p->faD[l] < 1 || p->faD[l] > 40) { delete p; return fail("filtact shifts"); }
      if (32767 * L1u + (1LL << (std::max(p->faA[l], p->faB[l]) - 1)) > INT32_MAX || 32767 * L1d + (1LL << (p->faE[l] - 1)) > INT32_MAX) { delete p; return fail("filtact int32 bound"); }
      sIn = ev;
    }
  }
  // LUTs
  std::vector<int> lut; const char *ln[4] = { "lut_exp2_frac_q15", "lut_log2_mant_q10", "lut_recip_mant_q15", "lut_sigmoid_q15" }; int *offs[4] = { &p->lutExp, &p->lutLog, &p->lutRec, &p->lutSig };
  const size_t need[4] = { 256, 1025, 1025, 8193 };
  for (int i = 0; i < 4; i++) { const Arr &a = A(ln[i]); if (a.v.size() < need[i]) { delete p; return fail("LUT size"); } *offs[i] = (int) lut.size(); for (long long v : a.v) lut.push_back((int) v); }

  // upload
  if (m) { this->~NNInterpCuda(); m = nullptr; }
  m = new Impl(); m->dev = device;
  auto ck = [&](cudaError_t e) { if (e != cudaSuccess) { snprintf(m_err, sizeof m_err, "CUDA: %s", cudaGetErrorString(e)); return false; } return true; };
  bool ok = ck(cudaSetDevice(device)) && ck(cudaStreamCreateWithFlags(&m->s, cudaStreamNonBlocking)) && ck(cudaMemcpyToSymbol(P, p, sizeof(DevP)))
            && ck(cudaMalloc(&m->dW3, w3.size() * sizeof(short))) && ck(cudaMemcpy(m->dW3, w3.data(), w3.size() * sizeof(short), cudaMemcpyHostToDevice))
            && ck(cudaMalloc(&m->dLut, lut.size() * sizeof(int))) && ck(cudaMemcpy(m->dLut, lut.data(), lut.size() * sizeof(int), cudaMemcpyHostToDevice))
            && ck(cudaMalloc(&m->dErr, sizeof(int))) && ck(cudaMallocHost(&m->hErr, sizeof(int))) && ck(cudaMemset(m->dErr, 0, sizeof(int)))
            && ck(cudaFuncSetAttribute(kFront, cudaFuncAttributeMaxDynamicSharedMemorySize, ARENA));
  delete p;
  m->paramBytes = w3.size() * sizeof(short) + lut.size() * sizeof(int) + sizeof(int);
  m_loaded = ok; return ok;
}

bool NNInterpCuda::predictBatch(const BlockIn *in, BlockOut *out, int n, int outMask)
{
  if (!m_loaded || n <= 0) return n == 0;
  auto ck = [&](cudaError_t e) { if (e != cudaSuccess) { snprintf(m_err, sizeof m_err, "CUDA: %s", cudaGetErrorString(e)); return false; } return true; };
  if (n > m->cap)
  {
    cudaFree(m->dIn); cudaFree(m->dOut); cudaFree(m->dPk[0]); cudaFree(m->dPk[1]); cudaFree(m->dY); cudaFreeHost(m->hY); m->dIn = nullptr; m->dOut = nullptr; m->dPk[0] = m->dPk[1] = nullptr; m->dY = m->hY = nullptr; m->cap = 0;
    int c = 64; while (c < n) c *= 2;
    if (!ck(cudaMalloc(&m->dIn, (size_t) c * sizeof(BlockIn))) || !ck(cudaMalloc(&m->dOut, (size_t) c * sizeof(BlockOut)))) return false;
    if (!ck(cudaMalloc(&m->dPk[0], (size_t) c * PKW * sizeof(unsigned))) || !ck(cudaMalloc(&m->dPk[1], (size_t) c * PKW * sizeof(unsigned)))) return false;
    if (!ck(cudaMalloc(&m->dY, (size_t) c * 256 * sizeof(long long))) || !ck(cudaMallocHost(&m->hY, (size_t) c * 256 * sizeof(long long)))) return false;
    m->cap = c;
  }
  if (!ck(cudaMemcpyAsync(m->dIn, in, (size_t) n * sizeof(BlockIn), cudaMemcpyHostToDevice, m->s))) return false;
  static const bool tim = getenv("NNI_CUDA_TIMING") != nullptr;   // yvvyee: per-kernel event timing (diagnostic)
  static cudaEvent_t ev[7]; static double acc[6]; static long cnt = 0; if (tim && !cnt) for (int i = 0; i < 7; i++) cudaEventCreate(&ev[i]);
  if (tim) cudaEventRecord(ev[0], m->s);
  kFront<<<n, NT, ARENA, m->s>>>(m->dIn, m->dOut, n, m->dPk[0], m->dW3, m->dLut, outMask, m->dErr);
  if (tim) cudaEventRecord(ev[1], m->s);
  kConvFA<6><<<dim3(16 / OPB, n), NTC, L_END, m->s>>>(0, m->dPk[0], m->dPk[1], m->dIn, m->dW3, n);
  if (tim) cudaEventRecord(ev[2], m->s);
  for (int l = 1; l < 4; l++) { kConvFA<8><<<dim3(16 / OPB, n), NTC, L_END, m->s>>>(l, m->dPk[l & 1], m->dPk[(l + 1) & 1], m->dIn, m->dW3, n); if (tim) cudaEventRecord(ev[2 + l], m->s); }
  const bool compact = (outMask & 7) == OUT_Y;   // yvvyee: y only: contiguous copy-back through a dense staging buffer
  kConv4L8<<<n, NTC, L8_END, m->s>>>(m->dPk[0], m->dIn, m->dOut, m->dW3, m->dLut, n, compact ? m->dY : nullptr);
  if (tim) { cudaEventRecord(ev[6], m->s); cudaEventSynchronize(ev[6]); for (int i = 0; i < 6; i++) { float ms; cudaEventElapsedTime(&ms, ev[i], ev[i + 1]); acc[i] += ms; } cnt++;
    if (cnt % 200 == 0) fprintf(stderr, "NNI_CUDA_TIMING n=%d us/batch: front %.1f conv0FA %.1f conv1FA %.1f conv2FA %.1f conv3FA %.1f conv4L8 %.1f\n", n, 1e3 * acc[0] / cnt, 1e3 * acc[1] / cnt, 1e3 * acc[2] / cnt, 1e3 * acc[3] / cnt, 1e3 * acc[4] / cnt, 1e3 * acc[5] / cnt); }
  if (!ck(cudaGetLastError())) return false;
  // yvvyee: copy back only the requested fields (y, ylin, xd are consecutive in BlockOut)
  const size_t offY = offsetof(BlockOut, y), offL = offsetof(BlockOut, ylin), offX = offsetof(BlockOut, xd);
  const size_t lo = (outMask & 1) ? offY : (outMask & 2) ? offL : offX;
  const size_t hi = (outMask & 4) ? sizeof(BlockOut) : (outMask & 2) ? offX : offL;
  if (compact)
  {
    if (!ck(cudaMemcpyAsync(m->hY, m->dY, (size_t) n * 256 * sizeof(long long), cudaMemcpyDeviceToHost, m->s))) return false;
  }
  else if (outMask & 7)
  {
    if (!ck(cudaMemcpy2DAsync((char *) out + lo, sizeof(BlockOut), (char *) m->dOut + lo, sizeof(BlockOut), hi - lo, n, cudaMemcpyDeviceToHost, m->s))) return false;
  }
  if (!ck(cudaMemcpyAsync(m->hErr, m->dErr, sizeof(int), cudaMemcpyDeviceToHost, m->s))) return false;
  if (!ck(cudaStreamSynchronize(m->s))) return false;
  if (*m->hErr) { snprintf(m_err, sizeof m_err, "invalid tile geometry in batch"); cudaMemset(m->dErr, 0, sizeof(int)); return false; }
  if (compact) for (int i = 0; i < n; i++) memcpy(out[i].y, m->hY + (size_t) i * 256, sizeof(long long) * in[i].tw * in[i].th);
  return true;
}
#ifdef NNI_PROF
void nniProfRead(long long *h) { cudaMemcpyFromSymbol(h, g_prof, sizeof(long long) * 64); }   // yvvyee: profiling build only
#endif
