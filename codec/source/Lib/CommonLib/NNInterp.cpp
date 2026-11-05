/* yvvyee: NNInterp 구현 — model/nninterp/nninterp_test.cpp 의 커널과 동일 (비트 일치 시험기 공유). */
#include "NNInterp.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <climits>

typedef __int128 i128;
typedef std::vector<int64_t> V;

NNInterp &NNInterp::get() { static NNInterp inst; return inst; }
const NNInterp::Arr &NNInterp::A(const std::string &n) const { auto it = m_t.find(n); if (it == m_t.end()) { fprintf(stderr, "NNInterp: missing %s\n", n.c_str()); exit(1); } return it->second; }

bool NNInterp::load(const std::string &dir)
{
  FILE *mf = fopen((dir + "/manifest.txt").c_str(), "r"); if (!mf) return false;
  char name[256], dtype[32], fn[256]; int nd;
  while (fscanf(mf, "%255s %31s %d", name, dtype, &nd) == 3)
  {
    Arr a; int64_t cnt = 1;
    for (int i = 0; i < nd; i++) { long long d; if (fscanf(mf, "%lld", &d) != 1) { fclose(mf); return false; } a.shape.push_back(d); cnt *= d; }
    if (fscanf(mf, "%255s", fn) != 1) { fclose(mf); return false; }
    FILE *f = fopen((dir + "/" + fn).c_str(), "rb"); if (!f) { fclose(mf); return false; } a.v.resize(cnt);
    if (std::string(dtype) == "int16") { std::vector<int16_t> t(cnt); if (fread(t.data(), 2, cnt, f) != (size_t) cnt) { fclose(f); fclose(mf); return false; } for (int64_t i = 0; i < cnt; i++) a.v[i] = t[i]; }
    else { std::vector<int32_t> t(cnt); if (fread(t.data(), 4, cnt, f) != (size_t) cnt) { fclose(f); fclose(mf); return false; } for (int64_t i = 0; i < cnt; i++) a.v[i] = t[i]; }
    fclose(f); m_t[name] = a;
  }
  fclose(mf); m_loaded = !m_t.empty(); if (m_loaded) buildFast(); return m_loaded;   // yvvyee: 가속 경로 준비 (조건 불충족이면 predictRef)
}

// ---------------------------------------------------------------- 산술
static const int PIXF = 4, FIRF = 14;   // yvvyee: 창 크기는 타일 크기에서 정한다 (WS 상수 제거)
static const int64_t PIX_DEN = 1024LL << PIXF;   // 정규화 1.0 = 1024 화소 (2진)
static inline int64_t rshr(i128 x, int sh)   // floor(x / 2^sh + 0.5); sh<0 은 좌시프트
{
  if (sh <= 0) return (int64_t) (x << (-sh));
  if (sh >= 120) return 0;
  i128 h = (i128) 1 << (sh - 1); return (int64_t) ((x + h) >> sh);
}
static inline int64_t rshr(int64_t x, int sh) { if (sh <= 0) return x << (-sh); if (sh >= 63) return 0; return (x + ((int64_t) 1 << (sh - 1))) >> sh; }
static inline int64_t fdiv_round(i128 num, i128 den)   // floor(num/den + 0.5), den > 0
{
  i128 n2 = 2 * num + den, d2 = 2 * den;
  return (int64_t) ((n2 >= 0) ? n2 / d2 : -((-n2 + d2 - 1) / d2));
}
static inline int64_t fdiv_floor(i128 num, i128 den) { return (int64_t) ((num >= 0) ? num / den : -((-num + den - 1) / den)); }
static inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline int64_t clamp16(int64_t x) { return x < -32767 ? -32767 : (x > 32767 ? 32767 : x); }
static inline int64_t clamp32(int64_t x) { return x < -2147483647LL ? -2147483647LL : (x > 2147483647LL ? 2147483647LL : x); }
static inline int msb128(i128 x) { int m = -1; while (x > 0) { x >>= 1; m++; } return m; }

// 값 = num * 2^-sc (num > 0) 의 정규화: e = floor(log2(값)), 가수 Q10 (1024..2048)
static inline void normalize(i128 num, int sc, int &e, int &m10)
{
  const int msb = msb128(num); e = msb - sc;
  if (msb >= 10) m10 = (int) (((num << 10) + ((i128) 1 << (msb - 1))) >> msb);   // floor(num/2^msb * 1024 + 0.5)
  else if (msb >= 1) m10 = (int) (((num << 10) + ((i128) 1 << (msb - 1))) >> msb);
  else m10 = 1024;
}
static const std::vector<int64_t> *LUT_EXP2, *LUT_LOG2, *LUT_RECIP, *LUT_SIG; static int64_t LOG2E_Q, LN2_Q, INV30_Q;

// e^x, x = x8/256 (Q8.8) -> 결과 = t * 2^(n-15), (t = 2^f 표 Q0.15). 반환 (t, n)
static inline void expQ(int64_t x8, int64_t &t, int &n)
{
  int64_t t8 = rshr((i128) x8 * LOG2E_Q, 16); t8 = std::max<int64_t>(-86 * 256, std::min<int64_t>(86 * 256, t8));
  n = (int) fdiv_floor(t8, 256); const int f = (int) (t8 - 256LL * n); t = LUT_EXP2->at(f);
}
// ln(값), 값 = num * 2^-sc : log2 = e + T[m]/1024 (Q10) ; ln(Q10) = rshr((e*1024+T)*LN2_Q, 16)
static inline int64_t lnQ10(i128 num, int sc)
{
  int e, m10; normalize(num, sc, e, m10); const int64_t l2 = (int64_t) e * 1024 + LUT_LOG2->at(m10 - 1024);
  return rshr((i128) l2 * LN2_Q, 16);
}
// 1/값 = r15 * 2^-15 * 2^-e
static inline void recipQ(i128 num, int sc, int64_t &r15, int &e) { int m10; normalize(num, sc, e, m10); r15 = LUT_RECIP->at(m10 - 1024); }
static inline int64_t sigQ15(int64_t x8) { x8 = std::max<int64_t>(-4096, std::min<int64_t>(4096, x8)); return LUT_SIG->at(x8 + 4096); }

// 분리형 FIR (복제 패딩) : 입력 격자 2^sIn, 출력 격자 2^sOut, 탭 Q1.14, 각 패스 후 반올림
// yvvyee: 각 패스 결과를 ±lim 으로 포화 (모사기 sep_blur_q: 화소 Q.pix ±32767, 에너지 Q.energy ±2^31-1). 타일 시험에서 에너지 가로 패스 중간값이 int32 를 넘는 화소가 나와 추가
static void sepFirRep(const V &x, int H, int W, const NNInterp::Arr &h, V &y, int sIn, int sMid, int sOut, int64_t lim)
{
  auto sat = [lim](int64_t v) { return v < -lim ? -lim : (v > lim ? lim : v); };
  const int T = (int) h.v.size(), r = (T - 1) / 2; V t(H * W); y.assign(H * W, 0);
  for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) { i128 acc = 0; for (int k = 0; k < T; k++) acc += (i128) x[yy * W + clampi(xx + k - r, 0, W - 1)] * h.v[k]; t[yy * W + xx] = sat(rshr(acc, FIRF + sIn - sMid)); }
  for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) { i128 acc = 0; for (int k = 0; k < T; k++) acc += (i128) t[clampi(yy + k - r, 0, H - 1) * W + xx] * h.v[k]; y[yy * W + xx] = sat(rshr(acc, FIRF + sMid - sOut)); }
}
// 에너지: (a*b) 정규화 (a,b 는 Q10.4, 추가 분모 extraDen) 를 2^e 격자로
static inline int64_t energyQ(int64_t a, int64_t b, int e, int64_t extraDen)
{
  i128 num = (i128) a * b; if (e >= 0) num <<= e; i128 den = (i128) PIX_DEN * PIX_DEN * extraDen;
  if (e < 0) den <<= (-e);
  return clamp32(fdiv_round(num, den));
}

// 창 sinc 이동: 입력 X (H x W, Q10.4), 이동 (sx16, sy16) 1/16 단위, 영역 [r0, r0+ny) x [r0, r0+nx) -> ny x nx (yvvyee: 직사각형 일반화)
static const NNInterp *g_nn = nullptr;
static inline const NNInterp::Arr &A(const std::string &n) { return g_nn->A_(n); }
static inline int64_t S(const std::string &n, int i = 0) { return g_nn->S_(n, i); }
static void shiftCrop(const V &X, int H, int W, int sx16, int sy16, int r0, int ny, int nx, V &out)
{
  const NNInterp::Arr &K = A("fir_kaiser16x12"); V Xh(H * nx);
  for (int t = 0; t < nx; t++)
  {
    const int p16 = (r0 + t) * 16 + sx16; const int i0 = (int) fdiv_floor(p16, 16), f = p16 - 16 * i0;
    for (int yy = 0; yy < H; yy++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) X[yy * W + clampi(i0 + k - 5, 0, W - 1)] * K.v[f * 12 + k]; Xh[yy * nx + t] = rshr(acc, FIRF); }
  }
  out.assign(ny * nx, 0);
  for (int t = 0; t < ny; t++)
  {
    const int p16 = (r0 + t) * 16 + sy16; const int i0 = (int) fdiv_floor(p16, 16), f = p16 - 16 * i0;
    for (int xx = 0; xx < nx; xx++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) Xh[clampi(i0 + k - 5, 0, H - 1) * nx + xx] * K.v[f * 12 + k]; out[t * nx + xx] = rshr(acc, FIRF); }
  }
}

// 1x1 합성곱 (채널별 입력 스케일 지수 sIn[c]) -> 출력 격자 2^sOut, 편향 Q1.14
static void conv1x1(const std::vector<V> &x, const std::vector<int> &sIn, int HW, const std::string &nm, std::vector<V> &y, int &sOut, bool relu)
{
  const NNInterp::Arr &w = A(nm + "_w"), &b = A(nm + "_b_q14"); const int we = (int) S(nm + "_wexp"); sOut = (int) S(nm + "_oexp");
  const int Co = (int) w.shape[0], Ci = (int) w.shape[1]; y.assign(Co, V(HW));
  int SM = 0; for (int c = 0; c < Ci; c++) SM = std::max(SM, sIn[c]);                                   // 입력 최대 스케일 (int64 안전)
  std::vector<int> up(Ci); for (int c = 0; c < Ci; c++) up[c] = SM - sIn[c];
  const int64_t *wv = w.v.data();
  for (int o = 0; o < Co; o++)
  {
    const int64_t bo = (we + SM - 14 >= 0) ? (b.v[o] << (we + SM - 14)) : (b.v[o] >> (14 - we - SM));
    for (int p = 0; p < HW; p++)
    {
      int64_t acc = bo;
      for (int c = 0; c < Ci; c++) acc += (wv[o * Ci + c] * x[c][p]) << up[c];                              // 스케일 2^(we+SM)
      int64_t v = clamp16(rshr(acc, we + SM - sOut)); if (relu && v < 0) v = 0; y[o][p] = v;
    }
  }
}
// 3x3 합성곱 (반사 패딩 1), 채널별 입력 스케일. LeakyReLU 는 입력에 (분자 51, 분모 256) 로 전달됨: xNum[c][p] = x*256 (양) / x*51 (음)
static void conv3x3(const std::vector<V> &xNum, const std::vector<int> &sIn, int Hh, int Ww, const std::string &nm, std::vector<V> &y, int &sOut)
{
  const NNInterp::Arr &w = A(nm + "_w"), &b = A(nm + "_b_q14"); const int we = (int) S(nm + "_wexp"); sOut = (int) S(nm + "_oexp");
  const int Co = (int) w.shape[0], Ci = (int) w.shape[1]; y.assign(Co, V(Hh * Ww));
  int SM = 0; for (int c = 0; c < Ci; c++) SM = std::max(SM, sIn[c] + 8);                                // 분자(/256) 포함 최대 스케일
  std::vector<int> up(Ci); for (int c = 0; c < Ci; c++) up[c] = SM - 8 - sIn[c];
  // 반사 패딩된 입력 (Hh+2)x(Ww+2) 를 미리 만들어 내부 루프에서 분기 제거
  const int Hp = Hh + 2, Wp = Ww + 2; std::vector<V> xp(Ci, V(Hp * Wp));
  auto refl = [](int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); };
  for (int c = 0; c < Ci; c++) for (int yy = 0; yy < Hp; yy++) for (int xx = 0; xx < Wp; xx++) xp[c][yy * Wp + xx] = xNum[c][refl(yy - 1, Hh) * Ww + refl(xx - 1, Ww)] << up[c];
  const int64_t *wv = w.v.data();
  for (int o = 0; o < Co; o++)
  {
    const int64_t bo = (we + SM - 14 >= 0) ? (b.v[o] << (we + SM - 14)) : (b.v[o] >> (14 - we - SM)); const int sh = we + SM - sOut;
    for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++)
    {
      int64_t acc = bo;
      for (int c = 0; c < Ci; c++)
      {
        const int64_t *wc = wv + (o * Ci + c) * 9; const int64_t *r0 = &xp[c][yy * Wp + xx], *r1 = r0 + Wp, *r2 = r1 + Wp;
        acc += wc[0] * r0[0] + wc[1] * r0[1] + wc[2] * r0[2] + wc[3] * r1[0] + wc[4] * r1[1] + wc[5] * r1[2] + wc[6] * r2[0] + wc[7] * r2[1] + wc[8] * r2[2];
      }
      y[o][yy * Ww + xx] = clamp16(rshr(acc, sh));
    }
  }
}
// FiltAct: 2배 업샘플(0삽입) -> sep(hu, 키 h/v) -> lrelu -> sep(hd, 키 h/v) -> [::2]. 입력 격자 sIn, 출력 격자 sOut(=ev)
static void filtAct(const V &x, int Hh, int Ww, int sIn, const std::string &nm, V &y, int &sOut)
{
  const NNInterp::Arr &hu = A("fir_filtact_hu_half"), &hd = A("fir_filtact_hd"); const int eh = (int) S(nm + "_oexp", 0), ev = (int) S(nm + "_oexp", 1);
  const int H2 = 2 * Hh, W2 = 2 * Ww; V u(H2 * W2, 0); for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) u[(2 * yy) * W2 + 2 * xx] = x[yy * Ww + xx];
  auto refl = [](int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); };
  // sep(hu): 탭 h14*2 (hu_half 표 x2), 반사 패딩 (6 좌, 5 우): 인덱스 k-6
  V t(H2 * W2), a(H2 * W2);
  for (int yy = 0; yy < H2; yy++) for (int xx = 0; xx < W2; xx++) { int64_t acc = 0; for (int k = 0; k < 12; k++) acc += (int64_t) u[yy * W2 + refl(xx + k - 6, W2)] * (2 * hu.v[k]); t[yy * W2 + xx] = clamp16(rshr(acc, FIRF + sIn - eh)); }
  for (int yy = 0; yy < H2; yy++) for (int xx = 0; xx < W2; xx++) { int64_t acc = 0; for (int k = 0; k < 12; k++) acc += (int64_t) t[refl(yy + k - 6, H2) * W2 + xx] * (2 * hu.v[k]); a[yy * W2 + xx] = clamp16(rshr(acc, FIRF + eh - ev)); }
  // lrelu (분자 51 / 분모 256), 격자 ev
  V num(H2 * W2); for (int i = 0; i < H2 * W2; i++) num[i] = a[i] >= 0 ? a[i] * 256 : a[i] * 51;
  // sep(hd): 입력 num/256 at 2^ev
  for (int yy = 0; yy < H2; yy++) for (int xx = 0; xx < W2; xx++) { int64_t acc = 0; for (int k = 0; k < 12; k++) acc += (int64_t) num[yy * W2 + refl(xx + k - 6, W2)] * hd.v[k]; t[yy * W2 + xx] = clamp16(rshr(acc, FIRF + 8 + ev - eh)); }
  for (int yy = 0; yy < H2; yy++) for (int xx = 0; xx < W2; xx++) { int64_t acc = 0; for (int k = 0; k < 12; k++) acc += (int64_t) t[refl(yy + k - 6, H2) * W2 + xx] * hd.v[k]; a[yy * W2 + xx] = clamp16(rshr(acc, FIRF + eh - ev)); }
  y.assign(Hh * Ww, 0); for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) y[yy * Ww + xx] = a[(2 * yy) * W2 + 2 * xx]; sOut = ev;
}

// 지도 특징: 경계 플래그 스캔 (codemap_feats 와 동일), 결과 Q1.14 (12채널). yvvyee: Hh x Ww 창 (가로 스캔 길이 Ww, 세로 Hh)
static void codemapFeats(const std::vector<int> &cm, int Hh, int Ww, int qp, std::vector<V> &cf)
{
  cf.assign(12, V(Hh * Ww, 0));
  auto scan = [&](bool horiz) {
    const int n = horiz ? Ww : Hh, na = horiz ? Hh : Ww;   // n: 스캔 축 길이, na: 고정 축 개수
    for (int a = 0; a < na; a++)   // a: 고정 축 (행 또는 열)
    {
      int last = -1; int64_t vb = 0;
      for (int i = 0; i < n; i++)
      {
        const int p = horiz ? a * Ww + i : i * Ww + a; const int flag = horiz ? (cm[p] & 1) : ((cm[p] >> 1) & 1); const int bs = horiz ? ((cm[p] >> 2) & 3) : ((cm[p] >> 4) & 3);
        if (flag) { last = i; vb = bs; }
        const int64_t d = last >= 0 ? i - last : n; const int64_t dq = std::min<int64_t>(d, 8) * 2048;   // d/8 * 16384
        if (horiz) { cf[0][p] = dq; cf[4][p] = last >= 0 ? vb * 8192 : 0; } else { cf[2][p] = dq; cf[6][p] = last >= 0 ? vb * 8192 : 0; }
      }
      int nxt = n + 1; int64_t vf = 0;
      for (int i = n - 1; i >= 0; i--)
      {
        const int p = horiz ? a * Ww + i : i * Ww + a;
        const int64_t d = (nxt <= n) ? (nxt - 1 - i) : n; const int64_t dq = std::min<int64_t>(d, 8) * 2048;
        if (horiz) { cf[1][p] = dq; cf[5][p] = (nxt <= n) ? vf * 8192 : 0; } else { cf[3][p] = dq; cf[7][p] = (nxt <= n) ? vf * 8192 : 0; }
        const int flag = horiz ? (cm[p] & 1) : ((cm[p] >> 1) & 1); const int bs = horiz ? ((cm[p] >> 2) & 3) : ((cm[p] >> 4) & 3);
        if (flag) { nxt = i; vf = bs; }   // e >= x+1 중 최소: 현재 위치 플래그는 다음(작은) 인덱스에서 보임
      }
    }
  };
  scan(true); scan(false);
  for (int p = 0; p < Hh * Ww; p++)
  {
    const int valid = (cm[p] >> 14) & 1, qpr = (cm[p] >> 8) & 63;
    cf[8][p] = ((cm[p] >> 6) & 1) * 16384; cf[9][p] = ((cm[p] >> 7) & 1) * 16384;
    cf[10][p] = valid ? clamp16(fdiv_round((i128) (qpr - qp) * 16384, 12)) : 0; cf[11][p] = valid * 16384;
  }
}


// yvvyee: 가속 전 기준 구현 (2026-10-21 이전 predict 와 같은 본문)
void NNInterp::predictRef(const BlockIn &in, BlockOut &out) const
{
  g_nn = this; LUT_EXP2 = &A("lut_exp2_frac_q15").v; LUT_LOG2 = &A("lut_log2_mant_q10").v; LUT_RECIP = &A("lut_recip_mant_q15").v; LUT_SIG = &A("lut_sigmoid_q15").v;
  LOG2E_Q = S("const_log2e_q16"); LN2_Q = S("const_ln2_q16"); INV30_Q = S("const_inv30_q16");
  const int nb = 4; const int K = (int) A("cand_dx_q4").shape[0];
  const int st = in.st, qp = in.qp, fx = std::max(0, in.fx), fy = std::max(0, in.fy);
  const int imv = in.imv, merge = in.merge, dpoc = in.dpoc, w4 = in.w4, h4 = in.h4;
  // yvvyee: 타일 크기와 창 (th+16)x(tw+16). 영역: L5·L6 (th+4)x(tw+4) (창 6..), L7 정렬 창 (th+12)x(tw+12) (창 2..), 출력 th x tw (창 8..)
  const int tw = in.tw, th = in.th, Hh = th + 16, Ww = tw + 16, HW = Hh * Ww, Ly = th + 4, Lx = tw + 4, Cy = th + 12, Cx = tw + 12;
  if (tw < 4 || th < 4 || tw > 16 || th > 16 || (tw & 3) || (th & 3) || in.stride < Ww || in.stride * (Hh - 1) + Ww > 1024 || K > 64)
  {
    fprintf(stderr, "NNInterp: bad tile tw=%d th=%d stride=%d K=%d\n", tw, th, in.stride, K); exit(1);
  }
  // ---- 입력 창 Q10.4, 지도 (yvvyee: 행 간격 stride 에서 조밀 Hh x Ww 로)
  V x(HW); std::vector<int> cm(HW);
  for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) { x[yy * Ww + xx] = (int64_t) in.win[yy * in.stride + xx] << PIXF; cm[yy * Ww + xx] = in.map[yy * in.stride + xx]; }
  // ---- 부가 입력 sd (16, Q1.14)
  V sd(16, 0); sd[0] = fdiv_round((i128) qp * 16384, 63); sd[1 + clampi(imv, 0, 3)] = 16384;
  { const bool integ = (fx == 0 && fy == 0), half = !integ && (fx % 8 == 0) && (fy % 8 == 0), quar = !integ && !half && (fx % 4 == 0) && (fy % 4 == 0);
    sd[5 + (integ ? 0 : half ? 1 : quar ? 2 : 3)] = 16384; }
  sd[9] = merge ? 16384 : 0; sd[10] = clamp16(fdiv_round((i128) std::abs(dpoc) * 16384, 16)); sd[11] = S("tab_log2bsz_q14", clampi(w4, 1, 32) - 1); sd[12] = S("tab_log2bsz_q14", clampi(h4, 1, 32) - 1); sd[13 + clampi(st, 0, 2)] = 16384;
  std::vector<V> cf; codemapFeats(cm, Hh, Ww, qp, cf);
  // ---- L1 대역 분해 (Q10.4)
  V g0, g1, g2; sepFirRep(x, Hh, Ww, A("fir_gauss06"), g0, 0, 0, 0, 32767); sepFirRep(x, Hh, Ww, A("fir_gauss12"), g1, 0, 0, 0, 32767); sepFirRep(x, Hh, Ww, A("fir_gauss24"), g2, 0, 0, 0, 32767);
  V B[4]; B[0] = g2; for (int b = 1; b < 4; b++) B[b].resize(HW);
  for (int i = 0; i < HW; i++) { B[1][i] = g1[i] - g2[i]; B[2][i] = g0[i] - g1[i]; B[3][i] = x[i] - g0[i]; }
  // ---- L2 phi_s (에너지 격자 phi_b)
  int eSq[4], ePhi[4]; for (int b = 0; b < 4; b++) { eSq[b] = (int) S("energy_sq" + std::to_string(b) + "_exp"); ePhi[b] = (int) S("energy_phi" + std::to_string(b) + "_exp"); }
  V phi[4];
  for (int b = 0; b < 4; b++) { V sq(HW); for (int i = 0; i < HW; i++) sq[i] = energyQ(B[b][i], B[b][i], eSq[b], 1); sepFirRep(sq, Hh, Ww, A("fir_gauss10"), phi[b], eSq[b], ePhi[b], ePhi[b], 2147483647LL); }
  // ---- L3 잡음 쿼리: [sd(16), cf(12)] -> 1x1 (relu) -> 1x1 -> lphi (Q8.8) -> phi_n (에너지 phin)
  std::vector<V> xin(28, V(HW)); std::vector<int> sIn(28, 14);
  for (int c = 0; c < 16; c++) std::fill(xin[c].begin(), xin[c].end(), sd[c]);
  for (int c = 0; c < 12; c++) xin[16 + c] = cf[c];
  std::vector<V> h1, h2; int s1, s2; conv1x1(xin, sIn, HW, "conv_noise_0", h1, s1, true); conv1x1(h1, std::vector<int>(32, s1), HW, "conv_noise_2", h2, s2, false);
  const int ePhin = (int) S("energy_phin_exp");
  std::vector<V> lphi(nb, V(HW)), phin(nb, V(HW)), lpn(nb, V(HW));
  const int64_t INV30 = INV30_Q;
  for (int b = 0; b < nb; b++) for (int p = 0; p < HW; p++)
  {
    lphi[b][p] = clamp16(rshr(((i128) h2[b][p] << 8) - ((i128) 7680 << s2), s2));                              // Q8.8
    int64_t t; int n; expQ(lphi[b][p], t, n); phin[b][p] = clamp32(rshr(t, 15 - n - ePhin));                   // 에너지 격자
    // lpn = Q1.14( ln(phi_n + 2^-40) * INV30 ) ; 값 = phin/2^ePhin + 2^-40
    const int Kc = std::max(ePhin, 40); i128 num = ((i128) phin[b][p] << (Kc - ePhin)) + ((i128) 1 << (Kc - 40));
    const int64_t ln10 = lnQ10(num, Kc); lpn[b][p] = clamp16(rshr((i128) ln10 * INV30, 12));
  }
  // ---- L4 Wiener: G15 = rshr(phi_s * r15, ePhi + e) ; xd = rshr(sum G15*B, 15)
  V xd(HW);
  for (int p = 0; p < HW; p++)
  {
    i128 acc = 0;
    for (int b = 0; b < nb; b++)
    {
      const int Kc = std::max(std::max(ePhi[b], ePhin), 40);
      i128 den = ((i128) phi[b][p] << (Kc - ePhi[b])) + ((i128) phin[b][p] << (Kc - ePhin)) + ((i128) 1 << (Kc - 40));
      int64_t r15; int e; recipQ(den, Kc, r15, e);
      const int64_t G15 = clamp32(rshr((i128) phi[b][p] * r15, ePhi[b] + e));
      acc += (i128) G15 * B[b][p];
    }
    xd[p] = rshr(acc, 15);
  }
  // ---- L4b 반화소 저역통과 (축별)
  { const NNInterp::Arr &k7 = A("alt_kernel7"); const int64_t gate = S("alt_gate_q15"); const bool ax = (imv == 3 && fx == 8), ay = (imv == 3 && fy == 8);
    if (ax) { V lx(HW); for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) { i128 acc = 0; for (int k = 0; k < 7; k++) acc += (i128) xd[yy * Ww + clampi(xx + k - 3, 0, Ww - 1)] * k7.v[k]; lx[yy * Ww + xx] = rshr(acc, FIRF); }
      for (int p = 0; p < HW; p++) xd[p] = rshr(((i128) xd[p] << 15) + (i128) gate * (lx[p] - xd[p]), 15); }
    if (ay) { V ly(HW); for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) { i128 acc = 0; for (int k = 0; k < 7; k++) acc += (i128) xd[clampi(yy + k - 3, 0, Hh - 1) * Ww + xx] * k7.v[k]; ly[yy * Ww + xx] = rshr(acc, FIRF); }
      for (int p = 0; p < HW; p++) xd[p] = rshr(((i128) xd[p] << 15) + (i128) gate * (ly[p] - xd[p]), 15); } }
  for (int i = 0; i < HW; i++) out.xd[i] = xd[i];
  // ---- L6 쿼리 입력: 구조 텐서 (gx, gy 반화소 단위: dq4/2 -> 제곱 분모 4), st0 격자 -> gauss10 -> st 격자, x1023, log1p 부호
  const int eSt0 = (int) S("energy_st0_exp"), eSt = (int) S("energy_st_exp");
  V gx(HW), gy(HW);
  for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) { gx[yy * Ww + xx] = xd[yy * Ww + clampi(xx + 1, 0, Ww - 1)] - xd[yy * Ww + clampi(xx - 1, 0, Ww - 1)]; gy[yy * Ww + xx] = xd[clampi(yy + 1, 0, Hh - 1) * Ww + xx] - xd[clampi(yy - 1, 0, Hh - 1) * Ww + xx]; }
  std::vector<V> lst(3, V(HW));
  for (int c = 0; c < 3; c++)
  {
    V s0(HW), s1v; for (int p = 0; p < HW; p++) s0[p] = energyQ(c == 0 ? gx[p] : (c == 1 ? gx[p] : gy[p]), c == 0 ? gx[p] : gy[p], eSt0, 4);
    sepFirRep(s0, Hh, Ww, A("fir_gauss10"), s1v, eSt0, eSt, eSt, 2147483647LL);
    for (int p = 0; p < HW; p++)
    {
      const int64_t v = s1v[p] * 1023; const int sg = v < 0 ? -1 : (v > 0 ? 1 : 0);        // |st| + 1 : 값 = |v|/2^eSt + 1
      i128 num = (i128) (v < 0 ? -v : v) + ((i128) 1 << eSt); const int64_t l10 = lnQ10(num, eSt);
      lst[c][p] = clamp16(rshr((i128) sg * l10, 2));                                           // Q8.8
    }
  }
  std::vector<V> qin; std::vector<int> qs;
  for (int c = 0; c < 16; c++) { qin.push_back(V(HW, sd[c])); qs.push_back(14); }
  for (int c = 0; c < 3; c++) { qin.push_back(lst[c]); qs.push_back(8); }
  qin.push_back(lpn[3]); qs.push_back(14); for (int c = 0; c < 4; c++) { qin.push_back(cf[c]); qs.push_back(14); }
  std::vector<V> q1, q; int sq1, sq; conv1x1(qin, qs, HW, "conv_dq_0", q1, sq1, true); conv1x1(q1, std::vector<int>(32, sq1), HW, "conv_dq_2", q, sq, false);
  // ---- L5 후보 이동 + L6 주의 ((th+4)x(tw+4), 창 6..)
  const NNInterp::Arr &dx = A("cand_dx_q4"), &dy = A("cand_dy_q4"), &phiM = A("mat_phi_i16"); const int ephi = (int) S("mat_phi_exp");
  std::vector<V> Vc(K); for (int c = 0; c < K; c++) shiftCrop(xd, Hh, Ww, fx + (int) dx.v[c], fy + (int) dy.v[c], 6, Ly, Lx, Vc[c]);
  V ylin(Ly * Lx);
  for (int t = 0; t < Ly * Lx; t++)
  {
    const int py = 6 + t / Lx, px = 6 + t % Lx, p = py * Ww + px; int64_t lg[64]; int64_t mx = INT64_MIN;
    for (int c = 0; c < K; c++) { i128 acc = 0; for (int j = 0; j < 5; j++) acc += (i128) q[j][p] * phiM.v[c * 5 + j]; lg[c] = rshr(acc, sq + ephi - 8); mx = std::max(mx, lg[c]); }
    int64_t e45[64]; i128 sum = 0;
    for (int c = 0; c < K; c++) { int64_t tt; int n; expQ(lg[c] - mx, tt, n); e45[c] = rshr(tt, 15 - n - 45); sum += e45[c]; }
    int64_t r15; int e; recipQ(sum, 45, r15, e); i128 acc = 0;
    for (int c = 0; c < K; c++) { const int64_t a15 = clamp32(rshr((i128) e45[c] * r15, 45 + e)); acc += (i128) a15 * Vc[c][t]; }
    ylin[t] = rshr(acc, 15);
  }
  for (int i = 0; i < Ly * Lx; i++) out.ylin[i] = ylin[i];
  // ---- L7 보정: 정렬 창 (th+12)x(tw+12) (창 2..) -> corr_in 격자, lpn 4채널, cf[0,1,2,3,8,9] 같은 영역 -> conv/FiltAct x4 -> conv -> 타일 영역 [6, 6+t)
  const int CHW = Cy * Cx;
  V Aw; shiftCrop(xd, Hh, Ww, fx, fy, 2, Cy, Cx, Aw); const int eIn = (int) S("corr_in_exp");
  std::vector<V> cin(11, V(CHW)); std::vector<int> cs(11, 14); cs[0] = eIn;
  for (int i = 0; i < CHW; i++) cin[0][i] = clamp16(fdiv_round(((i128) Aw[i]) << eIn, PIX_DEN));
  auto crop = [&](const V &src, V &dst) { dst.resize(CHW); for (int yy = 0; yy < Cy; yy++) for (int xx = 0; xx < Cx; xx++) dst[yy * Cx + xx] = src[(yy + 2) * Ww + xx + 2]; };
  for (int b = 0; b < 4; b++)
  {
    crop(lpn[b], cin[1 + b]);
  }
  const int cfi[6] = { 0, 1, 2, 3, 8, 9 };
  for (int k = 0; k < 6; k++)
  {
    crop(cf[cfi[k]], cin[5 + k]);
  }
  // 분자 형식 (분모 256): 첫 층 입력은 선형이므로 x*256
  std::vector<V> num(11, V(CHW)); for (int c = 0; c < 11; c++) for (int i = 0; i < CHW; i++) num[c][i] = cin[c][i] * 256;
  std::vector<V> c0, f0, c1, f1, c2, f2, c3, f3, c4; int sc0, sf0, sc1, sf1, sc2, sf2, sc3, sf3, sc4;
  conv3x3(num, cs, Cy, Cx, "conv_corr_0", c0, sc0);
  auto fa = [&](const std::vector<V> &cin_, int s_, const std::string &nm, std::vector<V> &out_, int &so_) { out_.resize(cin_.size()); for (size_t ch = 0; ch < cin_.size(); ch++) filtAct(cin_[ch], Cy, Cx, s_, nm, out_[ch], so_); };
  fa(c0, sc0, "filtact_corr_1", f0, sf0);
  auto tonum = [&](std::vector<V> &a, std::vector<V> &n) { n.assign(a.size(), V(CHW)); for (size_t ch = 0; ch < a.size(); ch++) for (int i = 0; i < CHW; i++) n[ch][i] = a[ch][i] * 256; };
  std::vector<V> n0; tonum(f0, n0); conv3x3(n0, std::vector<int>(32, sf0), Cy, Cx, "conv_corr_2", c1, sc1); fa(c1, sc1, "filtact_corr_3", f1, sf1);
  std::vector<V> n1; tonum(f1, n1); conv3x3(n1, std::vector<int>(32, sf1), Cy, Cx, "conv_corr_4", c2, sc2); fa(c2, sc2, "filtact_corr_5", f2, sf2);
  std::vector<V> n2; tonum(f2, n2); conv3x3(n2, std::vector<int>(32, sf2), Cy, Cx, "conv_corr_6", c3, sc3); fa(c3, sc3, "filtact_corr_7", f3, sf3);
  std::vector<V> n3; tonum(f3, n3); conv3x3(n3, std::vector<int>(32, sf3), Cy, Cx, "conv_corr_8", c4, sc4);
  // ---- L8 율 문턱 (yvvyee: 타일 안 4x4 블록마다): c = pix(D4 r D4^T), th(qp), arg = Q8.8(4(|c|-th) * recip(th)), c' = pix(c*sig), r' = pix(D4^T c' D4)
  const NNInterp::Arr &D = A("mat_d4_i16"); const int eD = (int) S("mat_d4_exp"); const int64_t thq = S("rate_th_q4_by_qp", clampi(qp, 0, 63));
  // recip(th): th 정규화 값 = th/PIX_DEN ; e: PIX_DEN*2^e <= th < PIX_DEN*2^(e+1) (QP 로만 정해지므로 블록 루프 밖에서 한 번)
  int eth = -60; while (((i128) PIX_DEN << (eth + 1 + 60)) <= ((i128) thq << 60)) eth++;   // e 는 음수일 수 있음
  int64_t r15th;
  { i128 numm = (i128) thq * 1024, denm = (i128) PIX_DEN; if (eth >= 0) denm <<= eth; else numm <<= (-eth);
    const int m10 = (int) fdiv_round(numm, denm); r15th = LUT_RECIP->at(m10 - 1024); }
  for (int by = 0; by < th; by += 4)
    for (int bx = 0; bx < tw; bx += 4)
    {
      V r(16); for (int yy = 0; yy < 4; yy++) for (int xx = 0; xx < 4; xx++) r[yy * 4 + xx] = c4[0][(by + yy + 6) * Cx + bx + xx + 6];   // 정규화 값 r/2^sc4
      V rq(16), c(16), cc(16);
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { i128 acc = 0; for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) acc += (i128) D.v[i * 4 + k] * r[k * 4 + l] * D.v[j * 4 + l]; c[i * 4 + j] = rshr(acc * PIX_DEN, 2 * eD + sc4); }
      for (int i = 0; i < 16; i++)
      {
        const int64_t ac = c[i] < 0 ? -c[i] : c[i];
        // arg = floor( 4*(|c|-th)/PIX_DEN * r15/2^15/2^e * 256 + 0.5 )
        i128 nn = (i128) 4 * (ac - thq) * r15th * 256; i128 dd = (i128) PIX_DEN << 15; if (eth >= 0) dd <<= eth; else nn <<= (-eth);
        const int64_t arg = clamp32(fdiv_round(nn, dd)); const int64_t s15 = sigQ15(arg);
        cc[i] = rshr((i128) c[i] * s15, 15);
      }
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { i128 acc = 0; for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) acc += (i128) D.v[k * 4 + i] * cc[k * 4 + l] * D.v[l * 4 + j]; rq[i * 4 + j] = rshr(acc, 2 * eD); }
      for (int yy = 0; yy < 4; yy++) for (int xx = 0; xx < 4; xx++) out.y[(by + yy) * tw + bx + xx] = ylin[(by + yy + 2) * Lx + bx + xx + 2] + rq[yy * 4 + xx];
    }
}

// ================================================================ 가속 경로 (yvvyee: 2026-10-21)
// yvvyee: predictRef 와 모든 출력 비트가 같도록 산술 순서·반올림·포화 지점은 그대로 두고 다음만 바꾼다.
//  (1) L7 3x3 합성곱: 채널 안쪽 int16 배치, 입력을 상위 바이트(x>>8, -128..127)와 하위 바이트(x&255, 0..255)로 나눠 AVX-VNNI(없으면 AVX2 madd) int32 누산 후 int64 로 합친다.
//      넘침 없음: |입력| <= 32767 (모든 생산자가 clamp16 또는 16384 이하), 출력 채널별 L1 = sum|w| 에 대해 255*L1 <= 2^31-1 을 적재 시 확인 (현재 최대 L1 1.89e6, 한계 8.42e6).
//  (2) FiltAct: 채널 fc (16 또는 32) 동시 처리, 0삽입 업샘플의 0 항(위상 절반) 생략, [::2] 로 버리는 출력 미계산, int32 누산.
//      넘침 없음: 입력 clamp16, 32767*sum|탭| + 반올림 2^(s-1) <= 2^31-1 을 적재 시 확인. lrelu(256/51) 뒤 가로 패스는 양·음 부분합 P,N 을 따로 int32 로 누산해 정확히 합친다.
//  (3) 2의 거듭제곱 분모 나눗셈은 반올림 시프트로, msb 는 clz 로, 블록 상수 부가 입력(sd)의 1x1 기여는 화소 루프 밖에서 한 번, 1x1 두 층은 화소별로 이어 계산.
//  (4) L5 가로 패스는 가로 이동이 같은 후보끼리 공유 (L7 정렬 창과도 공유), L6 쿼리(lst, 1x1)와 L7 마지막 층은 쓰이는 화소만 계산.
//  (5) i128 누산은 입력 최대 절댓값으로 int64 무넘침을 확인한 경우에만 int64 로 바꾸고, 아니면 원 i128 식을 쓴다.
//  적재 시 조건 하나라도 못 맞추면 m_fast 를 비워 predictRef 로 넘긴다. 타일 안에서 스레드를 만들지 않는다 (작업 버퍼는 thread_local).
#include <immintrin.h>
#include <cstring>

namespace {
struct C1L { int Co = 0, Ci = 0, we = 0, oexp = 0; const int64_t *w = nullptr, *b = nullptr; };
struct C3L { int Co = 0, Ci = 0, NP = 0, OB = 0, sh = 0; std::vector<int64_t> bo; std::vector<int32_t> wpk; };
constexpr int FCMAX = 32;       // yvvyee: FiltAct·중간 합성곱 채널 수 상한. 실제 채널 수는 적재 시 conv_corr_0 출력 채널에서 읽은 Fast::fc (16 또는 32)
constexpr int MAXHW = 1024, MAXC = 28 * 28, MAXH2 = 56;
struct Work
{
  int64_t x[MAXHW], g0[MAXHW], g1[MAXHW], g2[MAXHW], B[4][MAXHW], phi[4][MAXHW], tmp[MAXHW], sq[MAXHW], phin[4][MAXHW], lpn[4][MAXHW], xd[MAXHW], lxy[MAXHW];
  int64_t gx[MAXHW], gy[MAXHW], s1v[MAXHW], lst[3][MAXHW], cf[12][MAXHW], h2[8][MAXHW], q[8][MAXHW];
  int cm[MAXHW], pix[400];
  int64_t Xh[26][32 * 28], Vc[64][400], ylin[400], Aw[MAXC], c4[MAXC];
  int16_t act[MAXC * FCMAX], fa[MAXC * FCMAX], tA[28 * MAXH2 * FCMAX], t2[MAXH2 * 28 * FCMAX], aRow[MAXH2 * FCMAX], aPos[MAXH2 * FCMAX], aNeg[MAXH2 * FCMAX];
  uint32_t XH[30 * 30 * (FCMAX / 2)], XL[30 * 30 * (FCMAX / 2)];
};
Work &work() { thread_local std::unique_ptr<Work> w; if (!w) w.reset(new Work()); return *w; }
inline int reflI(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); }
inline int32_t clamp16i(int32_t x) { return x < -32767 ? -32767 : (x > 32767 ? 32767 : x); }
inline int msb128f(i128 x)   // yvvyee: msb128 과 같은 값 (x<=0 이면 -1)
{
  if (x <= 0) return -1;
  const uint64_t hi = (uint64_t) ((unsigned __int128) x >> 64), lo = (uint64_t) x;
  return hi ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll(lo);
}
inline void normalizeF(i128 num, int sc, int &e, int &m10)
{
  const int msb = msb128f(num); e = msb - sc;
  if (msb >= 1) m10 = (int) (((num << 10) + ((i128) 1 << (msb - 1))) >> msb); else m10 = 1024;
}
inline int64_t lnQ10F(i128 num, int sc) { int e, m10; normalizeF(num, sc, e, m10); const int64_t l2 = (int64_t) e * 1024 + LUT_LOG2->at(m10 - 1024); return rshr((i128) l2 * LN2_Q, 16); }
inline void recipQF(i128 num, int sc, int64_t &r15, int &e) { int m10; normalizeF(num, sc, e, m10); r15 = LUT_RECIP->at(m10 - 1024); }
inline void expQF(int64_t x8, int64_t &t, int &n)   // yvvyee: expQ 와 같은 값. floor(t8/256) 은 산술 시프트
{
  int64_t t8 = (x8 > -(1LL << 40) && x8 < (1LL << 40)) ? rshr(x8 * LOG2E_Q, 16) : rshr((i128) x8 * LOG2E_Q, 16);   // |x8*LOG2E_Q| < 2^40 * 2^18
  t8 = std::max<int64_t>(-86 * 256, std::min<int64_t>(86 * 256, t8)); n = (int) (t8 >> 8); t = LUT_EXP2->at((int) (t8 & 255));
}
// yvvyee: 분모 2^k 의 floor(num/2^k + 0.5) (fdiv_round(num, 2^k) 와 같다, k >= 1)
inline int64_t rdivPow2(i128 num, int k) { return (int64_t) ((num + ((i128) 1 << (k - 1))) >> k); }
// yvvyee: energyQ 와 같은 값. 분모 PIX_DEN^2*extraDen(=2^lgX)*2^max(0,-e) 는 2의 거듭제곱
inline int64_t energyQF(int64_t a, int64_t b, int e, int lgX)
{
  i128 num = (i128) a * b; if (e >= 0) num <<= e; return clamp32(rdivPow2(num, 28 + lgX + (e < 0 ? -e : 0)));
}
inline int64_t maxAbs(const int64_t *x, int n) { uint64_t m = 0; for (int i = 0; i < n; i++) { const uint64_t a = x[i] < 0 ? (uint64_t) 0 - (uint64_t) x[i] : (uint64_t) x[i]; m = a > m ? a : m; } return m > (uint64_t) INT64_MAX ? INT64_MAX : (int64_t) m; }

// yvvyee: sepFirRep 가속판. |x| < 2^31, lim < 2^31, sum|h| < 2^24, 1 <= 시프트 <= 40 이면 int64 누산이 정확 (|acc| < 2^31*2^24*... < 2^62), 아니면 원 함수
void sepFirF(const int64_t *x, int H, int W, const NNInterp::Arr &h, int64_t *y, int64_t *t, int sIn, int sMid, int sOut, int64_t lim)
{
  const int T = (int) h.v.size(), r = (T - 1) / 2, s1 = FIRF + sIn - sMid, s2 = FIRF + sMid - sOut; int64_t L1 = 0, hv[64];
  for (int k = 0; k < T && k < 64; k++) { hv[k] = h.v[k]; L1 += std::llabs(h.v[k]); }
  if (T > 64 || T > W || T > H || L1 >= (1LL << 24) || lim >= (1LL << 31) || s1 < 1 || s1 > 40 || s2 < 1 || s2 > 40 || maxAbs(x, H * W) >= (1LL << 31))
  {
    V xv(x, x + H * W), yv; sepFirRep(xv, H, W, h, yv, sIn, sMid, sOut, lim); std::copy(yv.begin(), yv.end(), y); return;
  }
  auto sat = [lim](int64_t v) { return v < -lim ? -lim : (v > lim ? lim : v); };
  const int64_t h1 = 1LL << (s1 - 1), h2 = 1LL << (s2 - 1);
  for (int yy = 0; yy < H; yy++)
  {
    const int64_t *xr = x + yy * W; int64_t *tr = t + yy * W;
    for (int xx = 0; xx < W; xx++)
    {
      int64_t acc = 0;
      if (xx >= r && xx + r < W) { const int64_t *s = xr + xx - r; for (int k = 0; k < T; k++) acc += s[k] * hv[k]; }
      else for (int k = 0; k < T; k++) acc += xr[clampi(xx + k - r, 0, W - 1)] * hv[k];
      tr[xx] = sat((acc + h1) >> s1);
    }
  }
  for (int yy = 0; yy < H; yy++)
  {
    int64_t acc[32] = { 0 };   // yvvyee: W <= 32
    for (int k = 0; k < T; k++) { const int64_t *s = t + clampi(yy + k - r, 0, H - 1) * W; const int64_t hk = hv[k]; for (int xx = 0; xx < W; xx++) acc[xx] += s[xx] * hk; }
    for (int xx = 0; xx < W; xx++) y[yy * W + xx] = sat((acc[xx] + h2) >> s2);
  }
}

// yvvyee: shiftCrop 의 가로 패스 (열 r0.., nx 개) / 세로 패스. 입력 |X| < 2^40 이면 int64 누산 정확 (sum|K| < 2^16, 12탭), 아니면 i128 (원식)
void hPassF(const int64_t *X, int H, int W, int sx16, int r0, int nx, const NNInterp::Arr &K, bool i64, int64_t *Xh)
{
  for (int t = 0; t < nx; t++)
  {
    const int p16 = (r0 + t) * 16 + sx16; const int i0 = (int) fdiv_floor(p16, 16), f = p16 - 16 * i0; const int64_t *kv = K.v.data() + f * 12;
    int idx[12]; for (int k = 0; k < 12; k++) idx[k] = clampi(i0 + k - 5, 0, W - 1);
    for (int yy = 0; yy < H; yy++)
    {
      const int64_t *xr = X + yy * W;
      if (i64) { int64_t acc = 0; for (int k = 0; k < 12; k++) acc += xr[idx[k]] * kv[k]; Xh[yy * nx + t] = rshr(acc, FIRF); }
      else { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) xr[idx[k]] * kv[k]; Xh[yy * nx + t] = rshr(acc, FIRF); }
    }
  }
}
// yvvyee: Xh (H x nxs, 열 오프셋 c0) 에서 세로 이동 sy16, 행 r0.., ny 개, nx 열 -> out (ny x nx)
void vPassF(const int64_t *Xh, int H, int nxs, int c0, int sy16, int r0, int ny, int nx, const NNInterp::Arr &K, bool i64, int64_t *out)
{
  for (int t = 0; t < ny; t++)
  {
    const int p16 = (r0 + t) * 16 + sy16; const int i0 = (int) fdiv_floor(p16, 16), f = p16 - 16 * i0; const int64_t *kv = K.v.data() + f * 12;
    const int64_t *rows[12]; for (int k = 0; k < 12; k++) rows[k] = Xh + clampi(i0 + k - 5, 0, H - 1) * nxs + c0;
    if (i64)
    {
      int64_t acc[32] = { 0 };
      for (int k = 0; k < 12; k++) { const int64_t *s = rows[k]; const int64_t kk = kv[k]; for (int xx = 0; xx < nx; xx++) acc[xx] += s[xx] * kk; }
      for (int xx = 0; xx < nx; xx++) out[t * nx + xx] = rshr(acc[xx], FIRF);
    }
    else for (int xx = 0; xx < nx; xx++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) rows[k][xx] * kv[k]; out[t * nx + xx] = rshr(acc, FIRF); }
  }
}

// yvvyee: codemapFeats 와 같은 값, 출력은 cf[12] 배열. cf[10] 은 qpr(0..63) 표로 (화소마다 i128 나눗셈 제거)
void codemapFeatsF(const int *cm, int Hh, int Ww, int qp, int64_t (*cf)[MAXHW])
{
  const int HW = Hh * Ww; for (int c = 0; c < 12; c++) std::fill(cf[c], cf[c] + HW, 0);
  auto scan = [&](bool horiz) {
    const int n = horiz ? Ww : Hh, na = horiz ? Hh : Ww;
    for (int a = 0; a < na; a++)
    {
      int last = -1; int64_t vb = 0;
      for (int i = 0; i < n; i++)
      {
        const int p = horiz ? a * Ww + i : i * Ww + a; const int flag = horiz ? (cm[p] & 1) : ((cm[p] >> 1) & 1); const int bs = horiz ? ((cm[p] >> 2) & 3) : ((cm[p] >> 4) & 3);
        if (flag) { last = i; vb = bs; }
        const int64_t d = last >= 0 ? i - last : n; const int64_t dq = std::min<int64_t>(d, 8) * 2048;
        if (horiz) { cf[0][p] = dq; cf[4][p] = last >= 0 ? vb * 8192 : 0; } else { cf[2][p] = dq; cf[6][p] = last >= 0 ? vb * 8192 : 0; }
      }
      int nxt = n + 1; int64_t vf = 0;
      for (int i = n - 1; i >= 0; i--)
      {
        const int p = horiz ? a * Ww + i : i * Ww + a;
        const int64_t d = (nxt <= n) ? (nxt - 1 - i) : n; const int64_t dq = std::min<int64_t>(d, 8) * 2048;
        if (horiz) { cf[1][p] = dq; cf[5][p] = (nxt <= n) ? vf * 8192 : 0; } else { cf[3][p] = dq; cf[7][p] = (nxt <= n) ? vf * 8192 : 0; }
        const int flag = horiz ? (cm[p] & 1) : ((cm[p] >> 1) & 1); const int bs = horiz ? ((cm[p] >> 2) & 3) : ((cm[p] >> 4) & 3);
        if (flag) { nxt = i; vf = bs; }
      }
    }
  };
  scan(true); scan(false);
  int64_t qtab[64]; for (int r = 0; r < 64; r++) qtab[r] = clamp16(fdiv_round((i128) (r - qp) * 16384, 12));
  for (int p = 0; p < HW; p++)
  {
    const int valid = (cm[p] >> 14) & 1, qpr = (cm[p] >> 8) & 63;
    cf[8][p] = ((cm[p] >> 6) & 1) * 16384; cf[9][p] = ((cm[p] >> 7) & 1) * 16384;
    cf[10][p] = valid ? qtab[qpr] : 0; cf[11][p] = valid * 16384;
  }
}

// yvvyee: 1x1 두 층 (a: relu, b: 선형) 을 화소별로 이어 계산 (화소 독립이라 원 conv1x1 두 번과 같다).
// 입력 채널 c: isC[c] 이면 블록 상수 x[c][0]. 원식 acc = bo + sum (w*x) << up[c] 를 상수 채널만 미리 더하고, 나머지는 (w << up) * x 로 (같은 정수).
// |x| <= 32767 (생산자 clamp16 또는 16384 이하), |w << up| < 2^31 이면 곱은 int64 에서 정확, 합도 int64 범위 (원 구현도 int64 누산).
// yvvyee: 1x1 두 층의 AVX2 판 (화소 4개 = int64 4줄). 곱은 vpmuldq (int32 x int32 -> int64, 피연산자 모두 int32 범위), 반올림 시프트는 산술 시프트 항등식
// floor(x / 2^s) = ((x ^ m) >>> s) ^ m (m = 부호 마스크) 으로 정확히. clamp16 뒤 relu 는 [0, 32767] 포화와 같다.
__attribute__((target("avx2"))) inline __m256i rshr64v(__m256i x, int sh)
{
  x = _mm256_add_epi64(x, _mm256_set1_epi64x(1LL << (sh - 1))); const __m256i m = _mm256_cmpgt_epi64(_mm256_setzero_si256(), x);
  return _mm256_xor_si256(_mm256_srl_epi64(_mm256_xor_si256(x, m), _mm_cvtsi32_si128(sh)), m);
}
__attribute__((target("avx2"))) inline __m256i clamp64v(__m256i v, int64_t lo, int64_t hi)
{
  const __m256i L = _mm256_set1_epi64x(lo), H = _mm256_set1_epi64x(hi);
  v = _mm256_blendv_epi8(v, L, _mm256_cmpgt_epi64(L, v)); return _mm256_blendv_epi8(v, H, _mm256_cmpgt_epi64(v, H));
}
__attribute__((target("avx2"))) void conv1x1x2Avx2(int Ca, int Cb, int nc, const int *cidx, const int64_t *baseA, const int32_t *wsa, int sha, const int64_t *baseB, const int32_t *wsb, int shb,
  const int64_t *const *x, const int *pix, int np, int64_t (*y)[MAXHW])
{
  thread_local __m256i WA[64 * 32], WB[32 * 8];   // yvvyee: 가중치를 한 번만 펼쳐 둔다 (화소 묶음마다 재사용)
  for (int i = 0; i < nc * Ca; i++) WA[i] = _mm256_set1_epi64x(wsa[i]);
  for (int i = 0; i < Ca * Cb; i++) WB[i] = _mm256_set1_epi64x(wsb[i]);
  for (int i = 0; i < np; i += 4)
  {
    const int p = pix ? pix[i] : i; __m256i xv[32], h1[32];
    for (int k = 0; k < nc; k++) xv[k] = _mm256_loadu_si256((const __m256i *) (x[cidx[k]] + p));
    for (int o = 0; o < Ca; o++)
    {
      __m256i acc = _mm256_set1_epi64x(baseA[o]);
      for (int k = 0; k < nc; k++) acc = _mm256_add_epi64(acc, _mm256_mul_epi32(xv[k], WA[k * Ca + o]));
      h1[o] = clamp64v(rshr64v(acc, sha), 0, 32767);
    }
    for (int o = 0; o < Cb; o++)
    {
      __m256i acc = _mm256_set1_epi64x(baseB[o]);
      for (int c = 0; c < Ca; c++) acc = _mm256_add_epi64(acc, _mm256_mul_epi32(h1[c], WB[c * Cb + o]));
      _mm256_storeu_si256((__m256i *) (y[o] + p), clamp64v(rshr64v(acc, shb), -32767, 32767));
    }
  }
}
// yvvyee: vPassF 의 AVX2 판 (열 4개 = int64 4줄). |Xh| < 2^31 (호출부 확인), |K| < 2^15 이라 vpmuldq 곱이 정확하고 12탭 합 < 2^50
__attribute__((target("avx2"))) void vPassAvx2(const int64_t *Xh, int H, int nxs, int c0, int sy16, int r0, int ny, int nx, const NNInterp::Arr &K, int64_t *out)
{
  for (int t = 0; t < ny; t++)
  {
    const int p16 = (r0 + t) * 16 + sy16; const int i0 = (int) fdiv_floor(p16, 16), f = p16 - 16 * i0; const int64_t *kv = K.v.data() + f * 12;
    const int64_t *rows[12]; __m256i kk[12];
    for (int k = 0; k < 12; k++) { rows[k] = Xh + clampi(i0 + k - 5, 0, H - 1) * nxs + c0; kk[k] = _mm256_set1_epi64x(kv[k]); }
    for (int xx = 0; xx < nx; xx += 4)
    {
      __m256i acc = _mm256_setzero_si256();
      for (int k = 0; k < 12; k++) acc = _mm256_add_epi64(acc, _mm256_mul_epi32(_mm256_loadu_si256((const __m256i *) (rows[k] + xx)), kk[k]));
      _mm256_storeu_si256((__m256i *) (out + t * nx + xx), rshr64v(acc, FIRF));
    }
  }
}
// yvvyee: conv1x1x2Avx2 의 AVX-512 판 (NNI_SIMD 3). 4화소 묶음 둘 (pix[i], pix[i+4]) 을 zmm 두 절반에 실어 int64 8줄로. 곱 (vpmuldq)·합·반올림·포화는 AVX2 판과 같다:
// floor((x + 2^(s-1)) / 2^s) 는 vpsraq (산술 시프트) 로 직접, 1 <= s <= 62 는 호출부 조건. np 가 8의 배수가 아니면 마지막 묶음은 두 절반에 같은 4화소 (한 번만 저장).
#define NNI_T512 "avx2,avx512f,avx512bw,avx512vnni"
__attribute__((target(NNI_T512))) void conv1x1x2Z(int Ca, int Cb, int nc, const int *cidx, const int64_t *baseA, const int32_t *wsa, int sha, const int64_t *baseB, const int32_t *wsb, int shb,
  const int64_t *const *x, const int *pix, int np, int64_t (*y)[MAXHW])
{
  int64_t WA[64 * 32], WB[32 * 8];   // yvvyee: int64 로 두면 _mm512_set1_epi64 가 메모리 방송 피연산자로 접힌다
  for (int i = 0; i < nc * Ca; i++) WA[i] = wsa[i];
  for (int i = 0; i < Ca * Cb; i++) WB[i] = wsb[i];
  const __m128i ca = _mm_cvtsi32_si128(sha), cb = _mm_cvtsi32_si128(shb);
  const __m512i ha = _mm512_set1_epi64(1LL << (sha - 1)), hb = _mm512_set1_epi64(1LL << (shb - 1)), z = _mm512_setzero_si512(), p16 = _mm512_set1_epi64(32767), n16 = _mm512_set1_epi64(-32767);
  for (int i = 0; i < np; i += 8)
  {
    const int p0 = pix ? pix[i] : i, p1 = i + 4 < np ? (pix ? pix[i + 4] : i + 4) : p0; __m512i xv[32], h1[32];
    for (int k = 0; k < nc; k++) { const int64_t *s = x[cidx[k]]; xv[k] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) (s + p0))), _mm256_loadu_si256((const __m256i *) (s + p1)), 1); }
    for (int o = 0; o < Ca; o++)
    {
      __m512i acc = _mm512_set1_epi64(baseA[o]);
      for (int k = 0; k < nc; k++) acc = _mm512_add_epi64(acc, _mm512_mul_epi32(xv[k], _mm512_set1_epi64(WA[k * Ca + o])));
      h1[o] = _mm512_min_epi64(_mm512_max_epi64(_mm512_sra_epi64(_mm512_add_epi64(acc, ha), ca), z), p16);
    }
    for (int o = 0; o < Cb; o++)
    {
      __m512i acc = _mm512_set1_epi64(baseB[o]);
      for (int c = 0; c < Ca; c++) acc = _mm512_add_epi64(acc, _mm512_mul_epi32(h1[c], _mm512_set1_epi64(WB[c * Cb + o])));
      const __m512i v = _mm512_min_epi64(_mm512_max_epi64(_mm512_sra_epi64(_mm512_add_epi64(acc, hb), cb), n16), p16);
      _mm256_storeu_si256((__m256i *) (y[o] + p0), _mm512_castsi512_si256(v)); if (p1 != p0) _mm256_storeu_si256((__m256i *) (y[o] + p1), _mm512_extracti64x4_epi64(v, 1));
    }
  }
}
// yvvyee: pix 가 있으면 4개씩 연속 (pix[i+3] == pix[i]+3, np 는 4의 배수) 이어야 AVX2 판을 쓴다
void conv1x1x2F(int simd, const C1L &La, const C1L &Lb, const int64_t *const *x, const int *sIn, const bool *isC, const int *pix, int np, int64_t (*y)[MAXHW])
{
  const int Ca = La.Co, Ci = La.Ci, Cb = Lb.Co; int SM = 0; for (int c = 0; c < Ci; c++) SM = std::max(SM, sIn[c]);
  const int64_t *wa = La.w, *wb = Lb.w;
  int64_t baseA[32]; const int bsa = La.we + SM - 14, sha = La.we + SM - La.oexp;
  for (int o = 0; o < Ca; o++) baseA[o] = bsa >= 0 ? (La.b[o] << bsa) : (La.b[o] >> (-bsa));
  int nc = 0, cidx[64]; int32_t wsa[64 * 32];   // yvvyee: 입력 격자 sIn 은 14 또는 8 (L3·L6 호출부 고정) 이라 up <= 6, |w << up| <= 2^21 : int32. 곱·합은 int64
  for (int c = 0; c < Ci; c++)
  {
    const int up = SM - sIn[c];
    if (isC[c]) { for (int o = 0; o < Ca; o++) baseA[o] += (wa[o * Ci + c] * x[c][0]) << up; }
    else { cidx[nc] = c; for (int o = 0; o < Ca; o++) wsa[nc * Ca + o] = (int32_t) (wa[o * Ci + c] << up); nc++; }
  }
  const int s1 = La.oexp, SMb = s1, bsb = Lb.we + SMb - 14, shb = Lb.we + SMb - Lb.oexp;   // b 층 입력은 모두 격자 s1 (up = 0)
  int64_t baseB[8]; for (int o = 0; o < Cb; o++) baseB[o] = bsb >= 0 ? (Lb.b[o] << bsb) : (Lb.b[o] >> (-bsb));
  int32_t wsb[32 * 8]; for (int c = 0; c < Ca; c++) for (int o = 0; o < Cb; o++) wsb[c * Cb + o] = (int32_t) wb[o * Ca + c];
  bool v4 = simd >= 1 && (np & 3) == 0 && sha >= 1 && sha <= 62 && shb >= 1 && shb <= 62;
  if (v4 && pix) for (int i = 0; i < np; i += 4) if (pix[i + 3] != pix[i] + 3) { v4 = false; break; }
  if (v4 && simd >= 3) { conv1x1x2Z(Ca, Cb, nc, cidx, baseA, wsa, sha, baseB, wsb, shb, x, pix, np, y); return; }   // yvvyee: AVX-512 (2026-11-04)
  if (v4) { conv1x1x2Avx2(Ca, Cb, nc, cidx, baseA, wsa, sha, baseB, wsb, shb, x, pix, np, y); return; }
  for (int i = 0; i < np; i++)
  {
    const int p = pix ? pix[i] : i; int64_t acc[32];
    for (int o = 0; o < Ca; o++) acc[o] = baseA[o];
    for (int k = 0; k < nc; k++) { const int32_t v = (int32_t) x[cidx[k]][p]; const int32_t *wr = wsa + k * Ca; for (int o = 0; o < Ca; o++) acc[o] += (int64_t) wr[o] * v; }
    int64_t hb[8]; for (int o = 0; o < Cb; o++) hb[o] = baseB[o];
    for (int c = 0; c < Ca; c++)
    {
      int64_t v = clamp16(rshr(acc[c], sha)); if (v < 0) v = 0;   // relu
      if (v) { const int32_t *wr = wsb + c * Cb; for (int o = 0; o < Cb; o++) hb[o] += (int64_t) wr[o] * (int32_t) v; }
    }
    for (int o = 0; o < Cb; o++) y[o][p] = clamp16(rshr(hb[o], shb));
  }
}

// yvvyee: 반사 패딩 (Hh+2)x(Ww+2) 와 상·하위 바이트 채널 쌍 묶음. a: [Hh*Ww][Cs] int16, 채널 c >= Cs 는 0
void packPad(const int16_t *a, int Hh, int Ww, int Cs, int NP, uint32_t *XH, uint32_t *XL)
{
  const int Wp = Ww + 2;
  for (int py = 0; py < Hh + 2; py++)
  {
    const int sy = reflI(py - 1, Hh);
    for (int px = 0; px < Wp; px++)
    {
      const int16_t *s = a + (sy * Ww + reflI(px - 1, Ww)) * Cs; uint32_t *h = XH + (py * Wp + px) * NP, *l = XL + (py * Wp + px) * NP;
      for (int cp = 0; cp < NP; cp++)
      {
        const int v0 = 2 * cp < Cs ? s[2 * cp] : 0, v1 = 2 * cp + 1 < Cs ? s[2 * cp + 1] : 0;
        h[cp] = (uint32_t) (uint16_t) (int16_t) (v0 >> 8) | ((uint32_t) (uint16_t) (int16_t) (v1 >> 8) << 16);
        l[cp] = (uint32_t) (v0 & 255) | ((uint32_t) (v1 & 255) << 16);
      }
    }
  }
}
// yvvyee: packPad 의 AVX-512 판 (NNI_SIMD 3, Cs == 2*NP 일 때). 채널 쌍 묶음은 int16 배열 순서 그대로이므로 XH = (int16) (x >> 8) (vpsraw), XL = x & 255 를 원소별로.
// 패딩 행 py 의 안쪽 열 1..Ww 는 원천 행 reflI(py-1) 의 연속 구간 Ww*Cs 개, 가장자리 열 0, Ww+1 은 원천 열 1, Ww-2 (패딩 열 2, Ww-1) 복사.
__attribute__((target(NNI_T512))) void packPadZ(const int16_t *a, int Hh, int Ww, int Cs, int NP, uint32_t *XH, uint32_t *XL)
{
  if (Cs != 2 * NP) { packPad(a, Hh, Ww, Cs, NP, XH, XL); return; }
  const int Wp = Ww + 2, n = Ww * Cs; const __m512i m255 = _mm512_set1_epi16(255);
  for (int py = 0; py < Hh + 2; py++)
  {
    const int16_t *s = a + reflI(py - 1, Hh) * Ww * Cs; uint32_t *hr = XH + py * Wp * NP, *lr = XL + py * Wp * NP;
    int16_t *h = (int16_t *) (hr + NP), *l = (int16_t *) (lr + NP);
    for (int i = 0; i < n; i += 32)
    {
      const __mmask32 m = n - i >= 32 ? (__mmask32) 0xFFFFFFFFu : (__mmask32) ((1u << (n - i)) - 1);
      const __m512i v = _mm512_maskz_loadu_epi16(m, s + i);
      _mm512_mask_storeu_epi16(h + i, m, _mm512_srai_epi16(v, 8)); _mm512_mask_storeu_epi16(l + i, m, _mm512_and_si512(v, m255));
    }
    memcpy(hr, hr + 2 * NP, NP * 4); memcpy(lr, lr + 2 * NP, NP * 4);
    memcpy(hr + (Wp - 1) * NP, hr + (Ww - 1) * NP, NP * 4); memcpy(lr + (Wp - 1) * NP, lr + (Ww - 1) * NP, NP * 4);
  }
}
// yvvyee: 3x3 합성곱 마무리. 원식 acc = bo + sum w * (x*256) (up=0) ; sum w*x = 256*Sh + Sl
inline void conv3Finish(const C3L &L, const int32_t *Sh, const int32_t *Sl, int16_t *o16, int64_t *o64)
{
  if (o16) for (int o = 0; o < L.Co; o++) o16[o] = (int16_t) clamp16(rshr(L.bo[o] + (((int64_t) Sh[o] * 256 + Sl[o]) << 8), L.sh));
  else o64[0] = clamp16(rshr(L.bo[0] + (((int64_t) Sh[0] * 256 + Sl[0]) << 8), L.sh));
}
// yvvyee: 출력 영역 [y0, y0+ny) x [x0, x0+nx). o16: [(Y*ostr+X)*Co + o], o64: [Y*ostr+X] (채널 0)
// yvvyee: VNNI 판은 누산 사슬이 지연에 묶이므로 (화소당 사슬 8개) 화소 NPX 개를 한 번에, 상·하위 바이트는 따로 지나간다 (사슬 4*NPX 개)
template <int OB, int NPX> __attribute__((target("avx2,avxvnni"))) inline void conv3x3VnniPart(const C3L &L, const uint32_t *Xp, int Wp, int Y, int X, int32_t *S)
{
  const int NP = L.NP; const int32_t *wk = L.wpk.data(); __m256i acc[NPX][OB];
  for (int px = 0; px < NPX; px++) for (int ob = 0; ob < OB; ob++) acc[px][ob] = _mm256_setzero_si256();
  for (int t = 0; t < 9; t++)
  {
    const uint32_t *pp = Xp + ((Y + t / 3) * Wp + X + t % 3) * NP; const int32_t *wt = wk + t * NP * OB * 8;
    for (int cp = 0; cp < NP; cp++)
    {
      __m256i b[NPX]; for (int px = 0; px < NPX; px++) b[px] = _mm256_set1_epi32((int) pp[px * NP + cp]);
      for (int ob = 0; ob < OB; ob++) { const __m256i W = _mm256_loadu_si256((const __m256i *) (wt + (cp * OB + ob) * 8)); for (int px = 0; px < NPX; px++) acc[px][ob] = _mm256_dpwssd_avx_epi32(acc[px][ob], b[px], W); }
    }
  }
  for (int px = 0; px < NPX; px++) for (int ob = 0; ob < OB; ob++) _mm256_storeu_si256((__m256i *) (S + px * 8 * OB + 8 * ob), acc[px][ob]);
}
template <int OB> __attribute__((target("avx2,avxvnni"))) void conv3x3Vnni(const C3L &L, const uint32_t *XH, const uint32_t *XL, int Wp, int y0, int x0, int ny, int nx, int16_t *o16, int64_t *o64, int ostr)
{
  constexpr int NPX = OB >= 4 ? 3 : 4; alignas(32) int32_t Sh[NPX * 8 * OB], Sl[NPX * 8 * OB];
  for (int Y = y0; Y < y0 + ny; Y++)
  {
    int X = x0;
    for (; X + NPX <= x0 + nx; X += NPX)
    {
      conv3x3VnniPart<OB, NPX>(L, XH, Wp, Y, X, Sh); conv3x3VnniPart<OB, NPX>(L, XL, Wp, Y, X, Sl);
      for (int px = 0; px < NPX; px++) conv3Finish(L, Sh + px * 8 * OB, Sl + px * 8 * OB, o16 ? o16 + (Y * ostr + X + px) * L.Co : nullptr, o64 ? o64 + Y * ostr + X + px : nullptr);
    }
    for (; X < x0 + nx; X++)
    {
      conv3x3VnniPart<OB, 1>(L, XH, Wp, Y, X, Sh); conv3x3VnniPart<OB, 1>(L, XL, Wp, Y, X, Sl);
      conv3Finish(L, Sh, Sl, o16 ? o16 + (Y * ostr + X) * L.Co : nullptr, o64 ? o64 + Y * ostr + X : nullptr);
    }
  }
}
template <int OB> __attribute__((target("avx2"))) void conv3x3Avx2(const C3L &L, const uint32_t *XH, const uint32_t *XL, int Wp, int y0, int x0, int ny, int nx, int16_t *o16, int64_t *o64, int ostr)
{
  const int NP = L.NP; const int32_t *wk = L.wpk.data();
  for (int Y = y0; Y < y0 + ny; Y++) for (int X = x0; X < x0 + nx; X++)
  {
    __m256i ah[OB], al[OB]; for (int ob = 0; ob < OB; ob++) { ah[ob] = _mm256_setzero_si256(); al[ob] = _mm256_setzero_si256(); }
    for (int t = 0; t < 9; t++)
    {
      const int q = ((Y + t / 3) * Wp + X + t % 3) * NP; const uint32_t *ph = XH + q, *pl = XL + q; const int32_t *wt = wk + t * NP * OB * 8;
      for (int cp = 0; cp < NP; cp++)
      {
        const __m256i bh = _mm256_set1_epi32((int) ph[cp]), bl = _mm256_set1_epi32((int) pl[cp]);
        for (int ob = 0; ob < OB; ob++) { const __m256i W = _mm256_loadu_si256((const __m256i *) (wt + (cp * OB + ob) * 8)); ah[ob] = _mm256_add_epi32(ah[ob], _mm256_madd_epi16(bh, W)); al[ob] = _mm256_add_epi32(al[ob], _mm256_madd_epi16(bl, W)); }
      }
    }
    alignas(32) int32_t Sh[8 * OB], Sl[8 * OB];
    for (int ob = 0; ob < OB; ob++) { _mm256_store_si256((__m256i *) (Sh + 8 * ob), ah[ob]); _mm256_store_si256((__m256i *) (Sl + 8 * ob), al[ob]); }
    conv3Finish(L, Sh, Sl, o16 ? o16 + (Y * ostr + X) * L.Co : nullptr, o64 ? o64 + Y * ostr + X : nullptr);
  }
}
// yvvyee: SIMD 없는 CPU 용. 같은 묶음 자료에서 int64 로 직접 합한다 (값은 같다)
void conv3x3Scalar(const C3L &L, const uint32_t *XH, const uint32_t *XL, int Wp, int y0, int x0, int ny, int nx, int16_t *o16, int64_t *o64, int ostr)
{
  const int NP = L.NP, OB = L.OB; const int32_t *wk = L.wpk.data();
  for (int Y = y0; Y < y0 + ny; Y++) for (int X = x0; X < x0 + nx; X++)
  {
    int64_t sh[32] = { 0 }, sl[32] = { 0 };
    for (int t = 0; t < 9; t++)
    {
      const int q = ((Y + t / 3) * Wp + X + t % 3) * NP;
      for (int cp = 0; cp < NP; cp++)
      {
        const uint32_t h = XH[q + cp], l = XL[q + cp]; const int h0 = (int16_t) (h & 0xffff), h1 = (int16_t) (h >> 16), l0 = (int16_t) (l & 0xffff), l1 = (int16_t) (l >> 16);
        for (int o = 0; o < 8 * OB; o++) { const uint32_t w = (uint32_t) wk[((t * NP + cp) * OB) * 8 + o]; const int w0 = (int16_t) (w & 0xffff), w1 = (int16_t) (w >> 16); sh[o] += (int64_t) w0 * h0 + (int64_t) w1 * h1; sl[o] += (int64_t) w0 * l0 + (int64_t) w1 * l1; }
      }
    }
    int32_t Sh[32], Sl[32]; for (int o = 0; o < 8 * OB; o++) { Sh[o] = (int32_t) sh[o]; Sl[o] = (int32_t) sl[o]; }
    conv3Finish(L, Sh, Sl, o16 ? o16 + (Y * ostr + X) * L.Co : nullptr, o64 ? o64 + Y * ostr + X : nullptr);
  }
}
// yvvyee: AVX-512 (avx512f, avx512bw, avx512_vnni) 판 3x3 합성곱 (NNI_SIMD 3). 묶음 자료(XH, XL, wpk)와 곱·합 항은 conv3x3Vnni 와 같고 레지스터 폭만 512비트.
//  OB 2 (출력 16채널): zmm 하나 = 출력 16채널, OB 4: zmm 둘. 상·하위 바이트를 같은 가중치 적재로 한 번에 지나간다 (사슬 = 화소 수 x zmm 수 x 2).
//  OB 1 (출력 8채널 이하, 마지막 층): zmm 하위 절반 = 상위 바이트 누산, 상위 절반 = 하위 바이트 누산 (가중치 8줄을 두 절반에 복제).
//  각 줄의 int32 누산은 conv3x3Vnni 와 같은 곱의 합 (vpdpwssd 는 비포화 랩 덧셈, 최종 합 |.| <= 255*L1 <= 2^31-1 을 적재 시 확인) 이라 값이 같다.
//  마무리는 int64 8줄로 conv3Finish 와 같은 식 (bo + ((Sh*256 + Sl) << 8), floor((x + 2^(sh-1)) / 2^sh) = 산술 시프트, clamp16). 1 <= sh <= 62 가 아니면 conv3Finish.
// yvvyee: NNI_T512 은 위에서 정의
template <int OB, int RY, int RX> __attribute__((target(NNI_T512))) inline __attribute__((always_inline))
void conv3x3ZBlk(const C3L &L, const uint32_t *XH, const uint32_t *XL, int Wp, int Y, int X, int16_t *o16, int64_t *o64, int ostr, bool vfin)
{
  constexpr int NZ = OB == 1 ? 1 : OB / 2, NPX = RY * RX;
  const int NP = L.NP; const int32_t *wk = L.wpk.data();
  __m512i ah[NPX][NZ], al[NPX][NZ]; int off[NPX];
  for (int px = 0; px < NPX; px++) { off[px] = ((Y + px / RX) * Wp + X + px % RX) * NP; for (int z = 0; z < NZ; z++) { ah[px][z] = _mm512_setzero_si512(); al[px][z] = _mm512_setzero_si512(); } }
  for (int t = 0; t < 9; t++)
  {
    const int toff = ((t / 3) * Wp + t % 3) * NP; const int32_t *wt = wk + t * NP * OB * 8;
    for (int cp = 0; cp < NP; cp++)
    {
      if (OB == 1)
      {
        const __m512i W = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *) (wt + cp * 8)));
        for (int px = 0; px < NPX; px++)
        {
          const int q = off[px] + toff + cp; const __m512i b = _mm512_mask_set1_epi32(_mm512_set1_epi32((int) XH[q]), (__mmask16) 0xFF00, (int) XL[q]);
          ah[px][0] = _mm512_dpwssd_epi32(ah[px][0], b, W);
        }
      }
      else
      {
        __m512i W[NZ]; for (int z = 0; z < NZ; z++) W[z] = _mm512_loadu_si512((const void *) (wt + cp * OB * 8 + 16 * z));
        for (int px = 0; px < NPX; px++)
        {
          const int q = off[px] + toff + cp; const __m512i bh = _mm512_set1_epi32((int) XH[q]), bl = _mm512_set1_epi32((int) XL[q]);
          for (int z = 0; z < NZ; z++) { ah[px][z] = _mm512_dpwssd_epi32(ah[px][z], bh, W[z]); al[px][z] = _mm512_dpwssd_epi32(al[px][z], bl, W[z]); }
        }
      }
    }
  }
  if (OB >= 2 && vfin)   // yvvyee: o16 이고 Co == 8*OB
  {
    const __m128i shc = _mm_cvtsi32_si128(L.sh); const __m512i hr = _mm512_set1_epi64(1LL << (L.sh - 1)), lo = _mm512_set1_epi64(-32767), hi = _mm512_set1_epi64(32767);
    for (int px = 0; px < NPX; px++)
    {
      int16_t *d = o16 + ((Y + px / RX) * ostr + X + px % RX) * L.Co;
      for (int z = 0; z < NZ; z++) for (int hf = 0; hf < 2; hf++)
      {
        const __m512i h64 = _mm512_cvtepi32_epi64(hf ? _mm512_extracti64x4_epi64(ah[px][z], 1) : _mm512_castsi512_si256(ah[px][z]));
        const __m512i l64 = _mm512_cvtepi32_epi64(hf ? _mm512_extracti64x4_epi64(al[px][z], 1) : _mm512_castsi512_si256(al[px][z]));
        __m512i v = _mm512_slli_epi64(_mm512_add_epi64(_mm512_slli_epi64(h64, 8), l64), 8);
        v = _mm512_add_epi64(v, _mm512_loadu_si512((const void *) (L.bo.data() + 16 * z + 8 * hf)));
        v = _mm512_sra_epi64(_mm512_add_epi64(v, hr), shc); v = _mm512_min_epi64(_mm512_max_epi64(v, lo), hi);
        _mm_storeu_si128((__m128i *) (d + 16 * z + 8 * hf), _mm512_cvtepi64_epi16(v));
      }
    }
    return;
  }
  for (int px = 0; px < NPX; px++)
  {
    alignas(64) int32_t Sh[16 * NZ], Sl[16 * NZ]; const int32_t *pSl = Sl;
    if (OB == 1) { _mm512_store_si512((void *) Sh, ah[px][0]); pSl = Sh + 8; }   // yvvyee: 하위 절반 = 상위 바이트, 상위 절반 = 하위 바이트
    else for (int z = 0; z < NZ; z++) { _mm512_store_si512((void *) (Sh + 16 * z), ah[px][z]); _mm512_store_si512((void *) (Sl + 16 * z), al[px][z]); }
    const int YY = Y + px / RX, XX = X + px % RX;
    conv3Finish(L, Sh, pSl, o16 ? o16 + (YY * ostr + XX) * L.Co : nullptr, o64 ? o64 + YY * ostr + XX : nullptr);
  }
}
// yvvyee: 화소 묶음 RY x RX (OB 1·2: 2x4, OB 4: 1x4 — zmm 레지스터 32개 안), 나머지는 1x1
template <int OB> __attribute__((target(NNI_T512))) void conv3x3Z(const C3L &L, const uint32_t *XH, const uint32_t *XL, int Wp, int y0, int x0, int ny, int nx, int16_t *o16, int64_t *o64, int ostr)
{
  constexpr int RY = OB == 4 ? 1 : 2, RX = 4;
  const bool vfin = o16 && OB >= 2 && L.Co == 8 * OB && L.sh >= 1 && L.sh <= 62;
  int Y = y0;
  for (; Y + RY <= y0 + ny; Y += RY)
  {
    int X = x0;
    for (; X + RX <= x0 + nx; X += RX) conv3x3ZBlk<OB, RY, RX>(L, XH, XL, Wp, Y, X, o16, o64, ostr, vfin);
    for (; X < x0 + nx; X++) for (int r = 0; r < RY; r++) conv3x3ZBlk<OB, 1, 1>(L, XH, XL, Wp, Y + r, X, o16, o64, ostr, vfin);
  }
  for (; Y < y0 + ny; Y++) for (int X = x0; X < x0 + nx; X++) conv3x3ZBlk<OB, 1, 1>(L, XH, XL, Wp, Y, X, o16, o64, ostr, vfin);
}
void conv3x3F(int simd, const C3L &L, const uint32_t *XH, const uint32_t *XL, int Wp, int y0, int x0, int ny, int nx, int16_t *o16, int64_t *o64, int ostr)
{
  if (simd >= 3) { switch (L.OB) { case 1: conv3x3Z<1>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; case 2: conv3x3Z<2>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; case 4: conv3x3Z<4>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; default: break; } }   // yvvyee: AVX-512 (2026-11-04)
  if (simd == 2) { switch (L.OB) { case 1: conv3x3Vnni<1>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; case 2: conv3x3Vnni<2>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; case 4: conv3x3Vnni<4>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; default: break; } }
  if (simd >= 1) { switch (L.OB) { case 1: conv3x3Avx2<1>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; case 2: conv3x3Avx2<2>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; case 4: conv3x3Avx2<4>(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr); return; default: break; } }
  conv3x3Scalar(L, XH, XL, Wp, y0, x0, ny, nx, o16, o64, ostr);
}

// yvvyee: FiltAct fc 채널 동시 (x, y: [Hh*Ww][fc] int16). FCV 가 0 이 아니면 fc 를 컴파일 시 상수로 고정한다. 단계 A (hu 가로, 짝수 행·짝수 원천 열만), B (hu 세로, 짝수 원천 행만), C·D (lrelu 뒤 hd 가로, 짝수 열만), E (hd 세로, 짝수 행·열만)
struct FaP { int sA, sB, sD, sE; int32_t hu2[12], hd[12]; };
template <int FCV> inline __attribute__((always_inline)) void filtActBody(const FaP &P, const int16_t *x, int Hh, int Ww, int16_t *y, Work &w, int fc)
{
  const int FC = FCV ? FCV : fc;
  const int H2 = 2 * Hh, W2 = 2 * Ww; const int32_t hA = 1 << (P.sA - 1), hB = 1 << (P.sB - 1), hE = 1 << (P.sE - 1);
  const int sA = P.sA, sB = P.sB, sD = P.sD, sE = P.sE, sD8 = sD - 8; const int32_t mP = (1 << sD8) - 1, mN = (1 << sD) - 1, hD = 1 << (sD - 1);
  int16_t *tA = w.tA, *t2 = w.t2, *aRow = w.aRow;
  // A: tA[i][xx] = 원 t[2i][xx]
  for (int xx = 0; xx < W2; xx++)
  {
    int jj[12], n = 0; int32_t tp[12];
    for (int k = 0; k < 12; k++) { const int m = reflI(xx + k - 6, W2); if (!(m & 1)) { jj[n] = m >> 1; tp[n] = P.hu2[k]; n++; } }
    for (int i = 0; i < Hh; i++)
    {
      int32_t acc[FCMAX] = { 0 };
      for (int m = 0; m < n; m++) { const int16_t *s = x + (i * Ww + jj[m]) * FC; const int32_t t = tp[m]; for (int c = 0; c < FC; c++) acc[c] += s[c] * t; }
      int16_t *d = tA + (i * W2 + xx) * FC; for (int c = 0; c < FC; c++) d[c] = (int16_t) clamp16i((acc[c] + hA) >> sA);
    }
  }
  int cj[28][12]; for (int j = 0; j < Ww; j++) for (int k = 0; k < 12; k++) cj[j][k] = reflI(2 * j + k - 6, W2);
  for (int yy = 0; yy < H2; yy++)
  {
    // B: 원 a[yy][xx]
    int ii[12], n = 0; int32_t tp[12];
    for (int k = 0; k < 12; k++) { const int m = reflI(yy + k - 6, H2); if (!(m & 1)) { ii[n] = m >> 1; tp[n] = P.hu2[k]; n++; } }
    for (int xx = 0; xx < W2; xx++)
    {
      int32_t acc[FCMAX] = { 0 };
      for (int m = 0; m < n; m++) { const int16_t *s = tA + (ii[m] * W2 + xx) * FC; const int32_t t = tp[m]; for (int c = 0; c < FC; c++) acc[c] += s[c] * t; }
      int16_t *d = aRow + xx * FC; for (int c = 0; c < FC; c++) d[c] = (int16_t) clamp16i((acc[c] + hB) >> sB);
    }
    // C·D: 원 t[yy][2j] = clamp16(rshr(256*P + 51*N, sD)), P = sum max(a,0)*hd, N = sum min(a,0)*hd
    for (int j = 0; j < Ww; j++)
    {
      int32_t pp[FCMAX] = { 0 }, nn[FCMAX] = { 0 };
      for (int k = 0; k < 12; k++) { const int16_t *s = aRow + cj[j][k] * FC; const int32_t h = P.hd[k]; for (int c = 0; c < FC; c++) { const int32_t v = s[c]; pp[c] += (v > 0 ? v : 0) * h; nn[c] += (v < 0 ? v : 0) * h; } }
      int16_t *d = t2 + (yy * Ww + j) * FC;
      for (int c = 0; c < FC; c++)
      {
        // floor((256P + 51N + 2^(sD-1)) / 2^sD) = Ph + 51*Nh + floor(R / 2^sD), R = 256*Pl + 51*Nl + 2^(sD-1) < 52.5*2^sD < 2^31 (sD <= 25)
        const int32_t Ph = pp[c] >> sD8, Pl = pp[c] & mP, Nh = nn[c] >> sD, Nl = nn[c] & mN;
        d[c] = (int16_t) clamp16i(Ph + 51 * Nh + (((Pl << 8) + 51 * Nl + hD) >> sD));
      }
    }
  }
  // E: 원 a[2i][2j]
  for (int i = 0; i < Hh; i++)
  {
    int rr[12]; for (int k = 0; k < 12; k++) rr[k] = reflI(2 * i + k - 6, H2);
    for (int j = 0; j < Ww; j++)
    {
      int32_t acc[FCMAX] = { 0 };
      for (int k = 0; k < 12; k++) { const int16_t *s = t2 + (rr[k] * Ww + j) * FC; const int32_t h = P.hd[k]; for (int c = 0; c < FC; c++) acc[c] += s[c] * h; }
      int16_t *d = y + (i * Ww + j) * FC; for (int c = 0; c < FC; c++) d[c] = (int16_t) clamp16i((acc[c] + hE) >> sE);
    }
  }
}
// yvvyee: FiltAct AVX2 판. 두 원천 행(열) k, k' 의 16채널 int16 을 unpacklo/hi 로 엮어 탭 쌍 (t_k, t_k') 과 madd: 곱 낭비 없이 int32 누산.
// unpack 은 128비트 줄마다 채널 [0-3, 8-11] / [4-7, 12-15] 순서를 만들고, 마무리의 packs_epi32 가 원래 순서로 되돌린다 (clamp16 뒤라 포화 없음).
// 항·반올림·포화는 filtActBody 와 같고, 쌍 합 |s_k t_k + s_k' t_k'| 과 부분합은 적재 시 확인한 32767*sum|탭| 한계 안이다.
struct FaPairs { int n; const int16_t *a[6], *b[6]; __m256i T[6]; };
// yvvyee: NH = 16채널 절반의 개수 (fc = 16 이면 1, 32 이면 2). NH == 1 이면 l1, h1 은 건드리지 않는다
template <int NH> __attribute__((target("avx2"))) inline void faAcc(const FaPairs &q, int off, __m256i &l0, __m256i &h0, __m256i &l1, __m256i &h1)
{
  for (int m = 0; m < q.n; m++)
  {
    const __m256i a0 = _mm256_loadu_si256((const __m256i *) (q.a[m] + off)), b0 = _mm256_loadu_si256((const __m256i *) (q.b[m] + off));
    l0 = _mm256_add_epi32(l0, _mm256_madd_epi16(_mm256_unpacklo_epi16(a0, b0), q.T[m])); h0 = _mm256_add_epi32(h0, _mm256_madd_epi16(_mm256_unpackhi_epi16(a0, b0), q.T[m]));
    if (NH == 2)
    {
      const __m256i a1 = _mm256_loadu_si256((const __m256i *) (q.a[m] + off + 16)), b1 = _mm256_loadu_si256((const __m256i *) (q.b[m] + off + 16));
      l1 = _mm256_add_epi32(l1, _mm256_madd_epi16(_mm256_unpacklo_epi16(a1, b1), q.T[m])); h1 = _mm256_add_epi32(h1, _mm256_madd_epi16(_mm256_unpackhi_epi16(a1, b1), q.T[m]));
    }
  }
}
__attribute__((target("avx2"))) inline __m256i faRnd(__m256i v, __m256i h, int sh)   // clamp16((v + h) >> sh)
{
  return _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(_mm256_add_epi32(v, h), sh), _mm256_set1_epi32(-32767)), _mm256_set1_epi32(32767));
}
// yvvyee: 탭 목록 (원천 포인터, 탭) 을 쌍으로. 홀수 개면 마지막 짝은 0 탭 (같은 포인터, 곱 0)
__attribute__((target("avx2"))) inline void faMkPairs(FaPairs &q, const int16_t *const *src, const int32_t *tp, int n)
{
  q.n = (n + 1) / 2;
  for (int m = 0; m < q.n; m++)
  {
    const int k0 = 2 * m, k1 = 2 * m + 1; q.a[m] = src[k0]; q.b[m] = k1 < n ? src[k1] : src[k0]; const int32_t t1 = k1 < n ? tp[k1] : 0;
    q.T[m] = _mm256_set1_epi32((int32_t) (((uint32_t) tp[k0] & 0xffff) | ((uint32_t) t1 << 16)));
  }
}
// yvvyee: lrelu 뒤 가로 패스 결합. floor((256P + 51N + 2^(sD-1)) / 2^sD) = Ph + 51*Nh + ((256*Pl + 51*Nl + 2^(sD-1)) >> sD), 뒤 clamp16
__attribute__((target("avx2"))) inline __m256i faComb(__m256i p, __m256i nv, int sD, __m256i mP, __m256i mN, __m256i hD)
{
  const __m256i c51 = _mm256_set1_epi32(51);
  const __m256i Ph = _mm256_srai_epi32(p, sD - 8), Pl = _mm256_and_si256(p, mP), Nh = _mm256_srai_epi32(nv, sD), Nl = _mm256_and_si256(nv, mN);
  const __m256i R = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(Pl, 8), _mm256_mullo_epi32(Nl, c51)), hD);
  const __m256i v = _mm256_add_epi32(_mm256_add_epi32(Ph, _mm256_mullo_epi32(Nh, c51)), _mm256_srai_epi32(R, sD));
  return _mm256_min_epi32(_mm256_max_epi32(v, _mm256_set1_epi32(-32767)), _mm256_set1_epi32(32767));
}
// yvvyee: NH = 1 (fc 16, 화소당 16채널 절반 하나) 또는 2 (fc 32, 절반 둘)
template <int NH> __attribute__((target("avx2"))) void filtActAvx2(const FaP &P, const int16_t *x, int Hh, int Ww, int16_t *y, Work &w)
{
  constexpr int FC = 16 * NH;
  const int H2 = 2 * Hh, W2 = 2 * Ww; const int sA = P.sA, sB = P.sB, sD = P.sD, sE = P.sE, sD8 = sD - 8;
  const __m256i hA = _mm256_set1_epi32(1 << (sA - 1)), hB = _mm256_set1_epi32(1 << (sB - 1)), hE = _mm256_set1_epi32(1 << (sE - 1)), hD = _mm256_set1_epi32(1 << (sD - 1));
  const __m256i mP = _mm256_set1_epi32((1 << sD8) - 1), mN = _mm256_set1_epi32((1 << sD) - 1), z = _mm256_setzero_si256();
  int16_t *tA = w.tA, *t2 = w.t2;
  const int16_t *src[12]; int32_t tp[12]; FaPairs q;
  // A: tA[i][xx] = 원 t[2i][xx] (짝수 원천 열만)
  for (int xx = 0; xx < W2; xx++)
  {
    int jj[12], n = 0;
    for (int k = 0; k < 12; k++) { const int m = reflI(xx + k - 6, W2); if (!(m & 1)) { jj[n] = m >> 1; tp[n] = P.hu2[k]; n++; } }
    for (int i = 0; i < Hh; i++)
    {
      for (int m = 0; m < n; m++) src[m] = x + (i * Ww + jj[m]) * FC;
      faMkPairs(q, src, tp, n); __m256i l0 = z, h0 = z, l1 = z, h1 = z; faAcc<NH>(q, 0, l0, h0, l1, h1);
      int16_t *d = tA + (i * W2 + xx) * FC;
      _mm256_storeu_si256((__m256i *) d, _mm256_packs_epi32(faRnd(l0, hA, sA), faRnd(h0, hA, sA)));
      if (NH == 2) _mm256_storeu_si256((__m256i *) (d + 16), _mm256_packs_epi32(faRnd(l1, hA, sA), faRnd(h1, hA, sA)));
    }
  }
  int cj[28][12]; for (int j = 0; j < Ww; j++) for (int k = 0; k < 12; k++) cj[j][k] = reflI(2 * j + k - 6, W2);
  FaPairs qd; { const int16_t *dummy[12]; for (int k = 0; k < 12; k++) dummy[k] = nullptr; faMkPairs(qd, dummy, P.hd, 12); }
  for (int yy = 0; yy < H2; yy++)
  {
    int ii[12], n = 0;
    for (int k = 0; k < 12; k++) { const int m = reflI(yy + k - 6, H2); if (!(m & 1)) { ii[n] = m >> 1; tp[n] = P.hu2[k]; n++; } }
    for (int m = 0; m < n; m++) src[m] = tA + ii[m] * W2 * FC;
    faMkPairs(q, src, tp, n);
    for (int xx = 0; xx < W2; xx++)   // B (원천 행 포인터 + 열 오프셋)
    {
      __m256i l0 = z, h0 = z, l1 = z, h1 = z; faAcc<NH>(q, xx * FC, l0, h0, l1, h1);
      // yvvyee: lrelu 의 양·음 부분을 열마다 한 번 (D 에서 열 하나를 6번 읽음)
      const __m256i r0 = _mm256_packs_epi32(faRnd(l0, hB, sB), faRnd(h0, hB, sB)), r1 = _mm256_packs_epi32(faRnd(l1, hB, sB), faRnd(h1, hB, sB));
      int16_t *dp = w.aPos + xx * FC, *dn = w.aNeg + xx * FC;
      _mm256_storeu_si256((__m256i *) dp, _mm256_max_epi16(r0, z));
      if (NH == 2) _mm256_storeu_si256((__m256i *) (dp + 16), _mm256_max_epi16(r1, z));
      _mm256_storeu_si256((__m256i *) dn, _mm256_min_epi16(r0, z));
      if (NH == 2) _mm256_storeu_si256((__m256i *) (dn + 16), _mm256_min_epi16(r1, z));
    }
    for (int j = 0; j < Ww; j++)   // C·D
    {
      for (int m = 0; m < 6; m++) { qd.a[m] = w.aPos + cj[j][2 * m] * FC; qd.b[m] = w.aPos + cj[j][2 * m + 1] * FC; }
      __m256i pl0 = z, ph0 = z, pl1 = z, ph1 = z; faAcc<NH>(qd, 0, pl0, ph0, pl1, ph1);
      for (int m = 0; m < 6; m++) { qd.a[m] = w.aNeg + cj[j][2 * m] * FC; qd.b[m] = w.aNeg + cj[j][2 * m + 1] * FC; }
      __m256i nl0 = z, nh0 = z, nl1 = z, nh1 = z; faAcc<NH>(qd, 0, nl0, nh0, nl1, nh1);
      // floor((256P + 51N + 2^(sD-1)) / 2^sD) = Ph + 51*Nh + ((256*Pl + 51*Nl + 2^(sD-1)) >> sD)
      int16_t *d = t2 + (yy * Ww + j) * FC;
      _mm256_storeu_si256((__m256i *) d, _mm256_packs_epi32(faComb(pl0, nl0, sD, mP, mN, hD), faComb(ph0, nh0, sD, mP, mN, hD)));
      if (NH == 2) _mm256_storeu_si256((__m256i *) (d + 16), _mm256_packs_epi32(faComb(pl1, nl1, sD, mP, mN, hD), faComb(ph1, nh1, sD, mP, mN, hD)));
    }
  }
  for (int i = 0; i < Hh; i++)   // E (짝수 행·열만)
  {
    for (int k = 0; k < 12; k++) src[k] = t2 + reflI(2 * i + k - 6, H2) * Ww * FC;
    faMkPairs(q, src, P.hd, 12);
    for (int j = 0; j < Ww; j++)
    {
      __m256i l0 = z, h0 = z, l1 = z, h1 = z; faAcc<NH>(q, j * FC, l0, h0, l1, h1);
      int16_t *d = y + (i * Ww + j) * FC;
      _mm256_storeu_si256((__m256i *) d, _mm256_packs_epi32(faRnd(l0, hE, sE), faRnd(h0, hE, sE)));
      if (NH == 2) _mm256_storeu_si256((__m256i *) (d + 16), _mm256_packs_epi32(faRnd(l1, hE, sE), faRnd(h1, hE, sE)));
    }
  }
}
void filtActBase(const FaP &P, const int16_t *x, int Hh, int Ww, int16_t *y, Work &w, int fc)
{
  if (fc == 32) filtActBody<32>(P, x, Hh, Ww, y, w, fc); else if (fc == 16) filtActBody<16>(P, x, Hh, Ww, y, w, fc); else filtActBody<0>(P, x, Hh, Ww, y, w, fc);
}
// yvvyee: AVX2 판은 fc 16·32 만, 그 밖은 기본 판
// yvvyee: FiltAct AVX-512 판 (NNI_SIMD 3, fc 16·32). 단계·항·반올림·포화는 filtActAvx2 와 같고, 탭 쌍 곱 합은 vpdpwssd (= add(acc, madd), 비포화 랩 덧셈이라 같은 값).
//  fc 16: zmm 하나 = 이웃 두 화소 x 16채널 (256비트 절반이 화소 하나). fc 32: zmm 하나 = 화소 하나.
//  unpack/packs 는 128비트 줄마다 동작하므로 각 줄 안의 채널 순서 복원은 AVX2 판과 같다.
//  A (hu 가로): 0삽입 업샘플의 짝·홀 출력 열 2j, 2j+1 을 원천 행의 반사 패딩판 PX[q] = x[reflI(2q-6, W2)/2] (q = 0..Ww+5) 에서 읽는다.
//     출력 2j = sum_i PX[j+i] * hu2[2i], 출력 2j+1 = sum_i PX[j+i+1] * hu2[2i+1] (i = 0..5). 반사는 홀짝을 보존하므로 원식의 (m 짝수인 k) 항 집합과 같다.
//  C·D (lrelu 뒤 hd 가로): 출력 j 의 탭 k 원천 열은 reflI(2j+k-6, W2). 탭 쌍 (2p, 2p+1) 의 두 열을 엮어 둔 반사 패딩판 PI[q] (q = j+p) 에서 읽는다 (아래 B 주석).
#define NNI_FCPAD (28 + 6)
struct FaZ { int n; const int16_t *a[6], *b[6]; __m512i T[6]; };
__attribute__((target(NNI_T512))) inline __m512i faT2(int32_t t0, int32_t t1) { return _mm512_set1_epi32((int32_t) (((uint32_t) t0 & 0xffff) | ((uint32_t) t1 << 16))); }
__attribute__((target(NNI_T512))) inline void faAccZ(const FaZ &q, int off, __m512i &l, __m512i &h)
{
  for (int m = 0; m < q.n; m++)
  {
    const __m512i a = _mm512_loadu_si512((const void *) (q.a[m] + off)), b = _mm512_loadu_si512((const void *) (q.b[m] + off));
    l = _mm512_dpwssd_epi32(l, _mm512_unpacklo_epi16(a, b), q.T[m]); h = _mm512_dpwssd_epi32(h, _mm512_unpackhi_epi16(a, b), q.T[m]);
  }
}
__attribute__((target(NNI_T512))) inline __m512i faRndZ(__m512i v, __m512i h, __m128i sh)   // clamp16((v + h) >> sh)
{
  return _mm512_min_epi32(_mm512_max_epi32(_mm512_sra_epi32(_mm512_add_epi32(v, h), sh), _mm512_set1_epi32(-32767)), _mm512_set1_epi32(32767));
}
__attribute__((target(NNI_T512))) inline __m512i faCombZ(__m512i p, __m512i nv, __m128i sD, __m128i sD8, __m512i mP, __m512i mN, __m512i hD)
{
  const __m512i c51 = _mm512_set1_epi32(51);
  const __m512i Ph = _mm512_sra_epi32(p, sD8), Pl = _mm512_and_si512(p, mP), Nh = _mm512_sra_epi32(nv, sD), Nl = _mm512_and_si512(nv, mN);
  const __m512i R = _mm512_add_epi32(_mm512_add_epi32(_mm512_slli_epi32(Pl, 8), _mm512_mullo_epi32(Nl, c51)), hD);
  const __m512i v = _mm512_add_epi32(_mm512_add_epi32(Ph, _mm512_mullo_epi32(Nh, c51)), _mm512_sra_epi32(R, sD));
  return _mm512_min_epi32(_mm512_max_epi32(v, _mm512_set1_epi32(-32767)), _mm512_set1_epi32(32767));
}
__attribute__((target(NNI_T512))) inline void faMkPairsZ(FaZ &q, const int16_t *const *src, const int32_t *tp, int n)
{
  q.n = (n + 1) / 2;
  for (int m = 0; m < q.n; m++) { const int k0 = 2 * m, k1 = 2 * m + 1; q.a[m] = src[k0]; q.b[m] = k1 < n ? src[k1] : src[k0]; q.T[m] = faT2(tp[k0], k1 < n ? tp[k1] : 0); }
}
template <int FC> __attribute__((target(NNI_T512))) void filtAct512(const FaP &P, const int16_t *x, int Hh, int Ww, int16_t *y, Work &w)
{
  constexpr int PV = 32 / FC;   // yvvyee: zmm 하나에 든 화소 수
  const int H2 = 2 * Hh, W2 = 2 * Ww;
  const __m128i sA = _mm_cvtsi32_si128(P.sA), sB = _mm_cvtsi32_si128(P.sB), sD = _mm_cvtsi32_si128(P.sD), sD8 = _mm_cvtsi32_si128(P.sD - 8), sE = _mm_cvtsi32_si128(P.sE);
  const __m512i hA = _mm512_set1_epi32(1 << (P.sA - 1)), hB = _mm512_set1_epi32(1 << (P.sB - 1)), hE = _mm512_set1_epi32(1 << (P.sE - 1)), hD = _mm512_set1_epi32(1 << (P.sD - 1));
  const __m512i mP = _mm512_set1_epi32((1 << (P.sD - 8)) - 1), mN = _mm512_set1_epi32((1 << P.sD) - 1), z = _mm512_setzero_si512();
  thread_local int16_t PX[NNI_FCPAD * FCMAX], PI[2][NNI_FCPAD * 2 * FCMAX];   // yvvyee: PI [양/음][q][열 쌍 x 채널]
  const __m512i ilv = _mm512_set_epi16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);   // yvvyee: int16 2c <- c, 2c+1 <- 16+c
  int16_t *tA = w.tA, *t2 = w.t2; const int16_t *src[12]; int32_t tp[12]; FaZ q;
  // A: tA[i][xx] = 원 t[2i][xx]
  __m512i TA[3], TE[3], TO[3];   // yvvyee: fc 16 은 TA (하위 절반 짝 출력 탭, 상위 절반 홀 출력 탭), fc 32 는 TE (짝 출력) / TO (홀 출력)
  for (int p = 0; p < 3; p++)
  {
    TE[p] = faT2(P.hu2[4 * p], P.hu2[4 * p + 2]); TO[p] = faT2(P.hu2[4 * p + 1], P.hu2[4 * p + 3]);
    TA[p] = _mm512_inserti64x4(TE[p], _mm512_castsi512_si256(TO[p]), 1);
  }
  for (int i = 0; i < Hh; i++)
  {
    for (int qq = 0; qq < Ww + 6; qq++) memcpy(PX + qq * FC, x + (i * Ww + (reflI(2 * qq - 6, W2) >> 1)) * FC, FC * sizeof(int16_t));
    for (int j = 0; j < Ww; j++)
    {
      int16_t *d = tA + (i * W2 + 2 * j) * FC;
      if (FC == 16)
      {
        __m512i l = z, h = z;
        for (int p = 0; p < 3; p++) { const __m512i a = _mm512_loadu_si512((const void *) (PX + (j + 2 * p) * 16)), b = _mm512_loadu_si512((const void *) (PX + (j + 2 * p + 1) * 16));
          l = _mm512_dpwssd_epi32(l, _mm512_unpacklo_epi16(a, b), TA[p]); h = _mm512_dpwssd_epi32(h, _mm512_unpackhi_epi16(a, b), TA[p]); }
        _mm512_storeu_si512((void *) d, _mm512_packs_epi32(faRndZ(l, hA, sA), faRndZ(h, hA, sA)));
      }
      else
      {
        for (int od = 0; od < 2; od++)
        {
          __m512i l = z, h = z;
          for (int p = 0; p < 3; p++) { const __m512i a = _mm512_loadu_si512((const void *) (PX + (j + od + 2 * p) * 32)), b = _mm512_loadu_si512((const void *) (PX + (j + od + 2 * p + 1) * 32));
            l = _mm512_dpwssd_epi32(l, _mm512_unpacklo_epi16(a, b), od ? TO[p] : TE[p]); h = _mm512_dpwssd_epi32(h, _mm512_unpackhi_epi16(a, b), od ? TO[p] : TE[p]); }
          _mm512_storeu_si512((void *) (d + od * 32), _mm512_packs_epi32(faRndZ(l, hA, sA), faRndZ(h, hA, sA)));
        }
      }
    }
  }
  __m512i TD[6]; for (int p = 0; p < 6; p++) TD[p] = faT2(P.hd[2 * p], P.hd[2 * p + 1]);
  for (int yy = 0; yy < H2; yy++)
  {
    int n = 0;
    for (int k = 0; k < 12; k++) { const int m = reflI(yy + k - 6, H2); if (!(m & 1)) { src[n] = tA + (m >> 1) * W2 * FC; tp[n] = P.hu2[k]; n++; } }
    faMkPairsZ(q, src, tp, n);
    // yvvyee: B: 원 a[yy][xx], a[yy][xx+1] (xx 짝수) 의 양·음 부분을 열 쌍 엮음판 PI[s][q] (q = (xx+6)/2) 에. 짝수 int16 = 열 xx, 홀수 = 열 xx+1 이라
    //   C·D 의 탭 쌍 (hd[2p], hd[2p+1]) 원천 (열 2(j+p)-6, 2(j+p)-5) 이 PI[s][j+p] 하나로 읽혀 unpack 이 없다.
    //   fc 16: vpermw 로 (열 xx 채널 c, 열 xx+1 채널 c) 를 int16 2c, 2c+1 에 -> C·D 의 int32 줄 = 채널 순서.
    //   fc 32: zmm 둘 = unpacklo/hi(열 xx, 열 xx+1) (AVX2 판 faAcc 의 l/h 와 같은 배치, C·D 끝 packs 가 채널 순서를 되돌린다).
    for (int xx = 0; xx < W2; xx += 2)
    {
      if (FC == 16)
      {
        __m512i l = z, h = z; faAccZ(q, xx * 16, l, h);
        const __m512i r = _mm512_permutexvar_epi16(ilv, _mm512_packs_epi32(faRndZ(l, hB, sB), faRndZ(h, hB, sB))); const int o = ((xx + 6) >> 1) * 32;
        _mm512_storeu_si512((void *) (PI[0] + o), _mm512_max_epi16(r, z)); _mm512_storeu_si512((void *) (PI[1] + o), _mm512_min_epi16(r, z));
      }
      else
      {
        __m512i l0 = z, h0 = z, l1 = z, h1 = z; faAccZ(q, xx * 32, l0, h0); faAccZ(q, xx * 32 + 32, l1, h1);
        const __m512i re = _mm512_packs_epi32(faRndZ(l0, hB, sB), faRndZ(h0, hB, sB)), ro = _mm512_packs_epi32(faRndZ(l1, hB, sB), faRndZ(h1, hB, sB));
        const __m512i a = _mm512_unpacklo_epi16(re, ro), b = _mm512_unpackhi_epi16(re, ro); const int o = ((xx + 6) >> 1) * 64;
        _mm512_storeu_si512((void *) (PI[0] + o), _mm512_max_epi16(a, z)); _mm512_storeu_si512((void *) (PI[0] + o + 32), _mm512_max_epi16(b, z));
        _mm512_storeu_si512((void *) (PI[1] + o), _mm512_min_epi16(a, z)); _mm512_storeu_si512((void *) (PI[1] + o + 32), _mm512_min_epi16(b, z));
      }
    }
    // yvvyee: 반사 패딩 q < 3, q >= Ww+3. 열 홀짝 par 칸 = 열 reflI(2q+par-6, W2) (반사는 홀짝 보존이라 원천도 par 칸, 원천 q' = (c+6-par)/2 는 안쪽)
    for (int qq = 0; qq < Ww + 6; qq++)
    {
      if (qq == 3) qq = Ww + 3;
      for (int par = 0; par < 2; par++)
      {
        const int c = reflI(2 * qq + par - 6, W2), cq = (c + 6 - par) >> 1; const __mmask32 m = par ? (__mmask32) 0xAAAAAAAAu : (__mmask32) 0x55555555u;
        for (int s = 0; s < 2; s++) for (int v = 0; v < FC / 16; v++)
        {
          int16_t *d = PI[s] + qq * 2 * FC + 32 * v; const int16_t *sp = PI[s] + cq * 2 * FC + 32 * v;
          _mm512_storeu_si512((void *) d, _mm512_mask_mov_epi16(_mm512_loadu_si512((const void *) d), m, _mm512_loadu_si512((const void *) sp)));
        }
      }
    }
    if (FC == 16)   // C·D: 원 t[yy][2j], 화소 둘씩 (누산 사슬 4개)
    {
      for (int j = 0; j < Ww; j += 2)
      {
        __m512i p0 = z, n0 = z, p1 = z, n1 = z;
        for (int p = 0; p < 6; p++)
        {
          const int o = (j + p) * 32;
          p0 = _mm512_dpwssd_epi32(p0, _mm512_loadu_si512((const void *) (PI[0] + o)), TD[p]); n0 = _mm512_dpwssd_epi32(n0, _mm512_loadu_si512((const void *) (PI[1] + o)), TD[p]);
          p1 = _mm512_dpwssd_epi32(p1, _mm512_loadu_si512((const void *) (PI[0] + o + 32)), TD[p]); n1 = _mm512_dpwssd_epi32(n1, _mm512_loadu_si512((const void *) (PI[1] + o + 32)), TD[p]);
        }
        int16_t *d = t2 + (yy * Ww + j) * 16;   // yvvyee: faCombZ 가 clamp16 한 뒤라 vpmovdw (절단) 가 값을 바꾸지 않는다
        _mm256_storeu_si256((__m256i *) d, _mm512_cvtepi32_epi16(faCombZ(p0, n0, sD, sD8, mP, mN, hD))); _mm256_storeu_si256((__m256i *) (d + 16), _mm512_cvtepi32_epi16(faCombZ(p1, n1, sD, sD8, mP, mN, hD)));
      }
    }
    else
    {
      for (int j = 0; j < Ww; j++)
      {
        __m512i pl = z, ph = z, nl = z, nh = z;
        for (int p = 0; p < 6; p++)
        {
          const int o = (j + p) * 64;
          pl = _mm512_dpwssd_epi32(pl, _mm512_loadu_si512((const void *) (PI[0] + o)), TD[p]); ph = _mm512_dpwssd_epi32(ph, _mm512_loadu_si512((const void *) (PI[0] + o + 32)), TD[p]);
          nl = _mm512_dpwssd_epi32(nl, _mm512_loadu_si512((const void *) (PI[1] + o)), TD[p]); nh = _mm512_dpwssd_epi32(nh, _mm512_loadu_si512((const void *) (PI[1] + o + 32)), TD[p]);
        }
        _mm512_storeu_si512((void *) (t2 + (yy * Ww + j) * 32), _mm512_packs_epi32(faCombZ(pl, nl, sD, sD8, mP, mN, hD), faCombZ(ph, nh, sD, sD8, mP, mN, hD)));
      }
    }
  }
  for (int i = 0; i < Hh; i++)   // E: 원 a[2i][2j]
  {
    for (int k = 0; k < 12; k++) src[k] = t2 + reflI(2 * i + k - 6, H2) * Ww * FC;
    faMkPairsZ(q, src, P.hd, 12);
    for (int j = 0; j < Ww; j += PV)
    {
      __m512i l = z, h = z; faAccZ(q, j * FC, l, h);
      _mm512_storeu_si512((void *) (y + (i * Ww + j) * FC), _mm512_packs_epi32(faRndZ(l, hE, sE), faRndZ(h, hE, sE)));
    }
  }
}
void filtActF(int simd, const FaP &P, const int16_t *x, int Hh, int Ww, int16_t *y, Work &w, int fc)
{
  if (simd >= 3 && fc == 32) { filtAct512<32>(P, x, Hh, Ww, y, w); return; } if (simd >= 3 && fc == 16) { filtAct512<16>(P, x, Hh, Ww, y, w); return; }   // yvvyee: AVX-512 (2026-11-04)
  if (simd >= 1 && fc == 32) filtActAvx2<2>(P, x, Hh, Ww, y, w); else if (simd >= 1 && fc == 16) filtActAvx2<1>(P, x, Hh, Ww, y, w); else filtActBase(P, x, Hh, Ww, y, w, fc);
}
} // namespace

struct NNInterp::Fast
{
  int simd = 0;   // yvvyee: 0 스칼라, 1 AVX2, 2 AVX2 + AVX-VNNI, 3 AVX-512 (avx512f, avx512bw, avx512_vnni; 3x3 합성곱·FiltAct 만 512비트, 나머지는 AVX2 경로)
  int fc = 32;    // yvvyee: L7 중간 채널 수 (conv_corr_0 출력 = conv_corr_2..6 입력·출력 = conv_corr_8 입력). 16 또는 32
  const Arr *g06, *g12, *g24, *g10, *k12, *k7, *cdx, *cdy, *phiM, *D, *tabB, *rth;
  int K, eSq[4], ePhi[4], ePhin, eSt0, eSt, ephi, eD, eIn, sc4 = 0; int64_t gate;
  C1L n0, n2, q0, q2; C3L c3[5]; FaP fa[4];
};

void NNInterp::buildFast()
{
  m_fast.reset();
  static const char *req[] = { "fir_gauss06", "fir_gauss12", "fir_gauss24", "fir_gauss10", "fir_kaiser16x12", "alt_kernel7", "alt_gate_q15", "cand_dx_q4", "cand_dy_q4", "mat_phi_i16", "mat_phi_exp", "mat_d4_i16", "mat_d4_exp",
    "tab_log2bsz_q14", "rate_th_q4_by_qp", "fir_filtact_hu_half", "fir_filtact_hd", "corr_in_exp", "energy_phin_exp", "energy_st0_exp", "energy_st_exp", "lut_exp2_frac_q15", "lut_log2_mant_q10", "lut_recip_mant_q15", "lut_sigmoid_q15",
    "const_log2e_q16", "const_ln2_q16", "const_inv30_q16" };
  for (const char *n : req) if (!m_t.count(n)) return;
  for (int b = 0; b < 4; b++) if (!m_t.count("energy_sq" + std::to_string(b) + "_exp") || !m_t.count("energy_phi" + std::to_string(b) + "_exp")) return;
  auto F = std::make_shared<Fast>(); Fast &f = *F;
  // yvvyee: 가속 경로가 쓰는 조회표·상수 전역 (predictRef 는 호출마다 같은 값을 다시 넣는다)
  g_nn = this; LUT_EXP2 = &A("lut_exp2_frac_q15").v; LUT_LOG2 = &A("lut_log2_mant_q10").v; LUT_RECIP = &A("lut_recip_mant_q15").v; LUT_SIG = &A("lut_sigmoid_q15").v;
  LOG2E_Q = S("const_log2e_q16"); LN2_Q = S("const_ln2_q16"); INV30_Q = S("const_inv30_q16");
  f.g06 = &A("fir_gauss06"); f.g12 = &A("fir_gauss12"); f.g24 = &A("fir_gauss24"); f.g10 = &A("fir_gauss10"); f.k12 = &A("fir_kaiser16x12"); f.k7 = &A("alt_kernel7"); f.cdx = &A("cand_dx_q4"); f.cdy = &A("cand_dy_q4");
  f.phiM = &A("mat_phi_i16"); f.D = &A("mat_d4_i16"); f.tabB = &A("tab_log2bsz_q14"); f.rth = &A("rate_th_q4_by_qp");
  f.K = (int) f.cdx->shape[0]; f.ePhin = (int) S("energy_phin_exp"); f.eSt0 = (int) S("energy_st0_exp"); f.eSt = (int) S("energy_st_exp"); f.ephi = (int) S("mat_phi_exp"); f.eD = (int) S("mat_d4_exp"); f.eIn = (int) S("corr_in_exp"); f.gate = S("alt_gate_q15");
  for (int b = 0; b < 4; b++) { f.eSq[b] = (int) S("energy_sq" + std::to_string(b) + "_exp"); f.ePhi[b] = (int) S("energy_phi" + std::to_string(b) + "_exp"); }
  if (f.K < 1 || f.K > 25 || (int) f.cdy->v.size() < f.K || (int) f.phiM->v.size() < f.K * 5 || f.k12->v.size() != 16 * 12 || f.tabB->v.size() < 32 || f.rth->v.size() < 64 || f.D->v.size() < 16) return;
  for (int64_t v : f.k12->v) if (v < -32768 || v > 32767) return;   // yvvyee: hPassF/vPassF int64 무넘침 전제 (|K| < 2^15)
  // 1x1 층
  auto mk1 = [&](const std::string &nm, C1L &L) {
    const std::string wn = nm + "_w", bn = nm + "_b_q14", en = nm + "_wexp", on = nm + "_oexp"; if (!m_t.count(wn) || !m_t.count(bn) || !m_t.count(en) || !m_t.count(on)) return false;
    const Arr &w = A(wn); if (w.shape.size() < 2) return false; L.Co = (int) w.shape[0]; L.Ci = (int) w.shape[1]; L.we = (int) S(en); L.oexp = (int) S(on); L.w = w.v.data(); L.b = A(bn).v.data();
    if ((int64_t) w.v.size() != (int64_t) L.Co * L.Ci || (int) A(bn).v.size() < L.Co) return false;
    for (int64_t v : w.v) if (v < -32768 || v > 32767) return false;
    return true; };
  if (!mk1("conv_noise_0", f.n0) || !mk1("conv_noise_2", f.n2) || !mk1("conv_dq_0", f.q0) || !mk1("conv_dq_2", f.q2)) return;
  if (f.n0.Ci != 28 || f.n0.Co > 32 || f.n2.Ci != f.n0.Co || f.n2.Co < 4 || f.n2.Co > 8 || f.q0.Ci != 24 || f.q0.Co > 32 || f.q2.Ci != f.q0.Co || f.q2.Co < 5 || f.q2.Co > 8) return;
  // 3x3 층과 FiltAct
  const Arr &hu = A("fir_filtact_hu_half"), &hd = A("fir_filtact_hd"); if (hu.v.size() != 12 || hd.v.size() != 12) return;
  int64_t L1u = 0, L1d = 0; for (int k = 0; k < 12; k++) { L1u += std::llabs(2 * hu.v[k]); L1d += std::llabs(hd.v[k]); }
  const char *cn[5] = { "conv_corr_0", "conv_corr_2", "conv_corr_4", "conv_corr_6", "conv_corr_8" }, *fn[4] = { "filtact_corr_1", "filtact_corr_3", "filtact_corr_5", "filtact_corr_7" };
  int sIn = f.eIn;   // yvvyee: 첫 층 입력 격자 (cs: 0번 eIn, 나머지 14). eIn != 14 면 채널별 up 이 달라 가속 경로 불가
  if (f.eIn != 14) return;
  // yvvyee: 중간 채널 수는 첫 3x3 층 출력 채널에서 읽는다. 16·32 만 가속 경로 (그 밖은 predictRef)
  if (!m_t.count("conv_corr_0_w") || A("conv_corr_0_w").shape.size() != 4) return;
  f.fc = (int) A("conv_corr_0_w").shape[0]; if (f.fc != 16 && f.fc != 32) return;
  for (int l = 0; l < 5; l++)
  {
    const std::string nm = cn[l]; if (!m_t.count(nm + "_w") || !m_t.count(nm + "_b_q14") || !m_t.count(nm + "_wexp") || !m_t.count(nm + "_oexp")) return;
    const Arr &w = A(nm + "_w"), &b = A(nm + "_b_q14"); if (w.shape.size() != 4 || w.shape[2] != 3 || w.shape[3] != 3) return;
    C3L &L = f.c3[l]; L.Co = (int) w.shape[0]; L.Ci = (int) w.shape[1]; const int we = (int) S(nm + "_wexp"), so = (int) S(nm + "_oexp");
    if (l == 0 ? L.Ci != 11 : L.Ci != f.fc) return;
    if (l < 4 ? L.Co != f.fc : (L.Co < 1 || L.Co > 32)) return;
    if ((int) b.v.size() < L.Co) return;
    L.NP = (L.Ci + 1) / 2; L.OB = (L.Co + 7) / 8; if (L.OB == 3) L.OB = 4;
    const int SM = sIn + 8; L.sh = we + SM - so; L.bo.resize(L.Co);
    for (int o = 0; o < L.Co; o++) L.bo[o] = (we + SM - 14 >= 0) ? (b.v[o] << (we + SM - 14)) : (b.v[o] >> (14 - we - SM));
    L.wpk.assign((size_t) 9 * L.NP * L.OB * 8, 0);
    for (int o = 0; o < L.Co; o++)
    {
      int64_t l1 = 0;
      for (int c = 0; c < L.Ci; c++) for (int t = 0; t < 9; t++) { const int64_t v = w.v[(o * L.Ci + c) * 9 + t]; if (v < -32768 || v > 32767) return; l1 += std::llabs(v); }
      if (255 * l1 > INT32_MAX) return;   // yvvyee: 하위 바이트(0..255) 합 |sum w*xl| <= 255*L1, 상위 바이트 |sum w*xh| <= 128*L1
      for (int c = 0; c < L.Ci; c++) for (int t = 0; t < 9; t++)
      {
        const int cp = c >> 1, ob = o >> 3, ln = o & 7; uint32_t &dst = reinterpret_cast<uint32_t &>(L.wpk[((size_t) (t * L.NP + cp) * L.OB + ob) * 8 + ln]);
        const uint32_t v = (uint16_t) (int16_t) w.v[(o * L.Ci + c) * 9 + t]; dst |= (c & 1) ? (v << 16) : v;
      }
    }
    if (l == 4) f.sc4 = so;
    if (l < 4)
    {
      const std::string fnm = fn[l]; if (!m_t.count(fnm + "_oexp") || A(fnm + "_oexp").v.size() < 2) return;
      const int eh = (int) S(fnm + "_oexp", 0), ev = (int) S(fnm + "_oexp", 1); FaP &P = f.fa[l];
      P.sA = FIRF + so - eh; P.sB = FIRF + eh - ev; P.sD = FIRF + 8 + ev - eh; P.sE = FIRF + eh - ev;
      if (P.sA < 1 || P.sA > 30 || P.sB < 1 || P.sB > 30 || P.sE < 1 || P.sE > 30 || P.sD < 8 || P.sD > 25) return;
      if (32767 * L1u + (1LL << (std::max(P.sA, P.sB) - 1)) > INT32_MAX || 32767 * L1d + (1LL << (P.sE - 1)) > INT32_MAX) return;
      for (int k = 0; k < 12; k++) { P.hu2[k] = (int32_t) (2 * hu.v[k]); P.hd[k] = (int32_t) hd.v[k]; }
      sIn = ev;
    }
  }
  f.simd = 0;
  bool vnni256 = false;   // yvvyee: AVX-VNNI (256비트) 유무. AVX-512 는 있고 AVX-VNNI 는 없는 CPU 에서 NNI_SIMD=2 로 낮출 때 1 로 내린다
#if defined(__x86_64__) || defined(__i386__)
  __builtin_cpu_init(); if (__builtin_cpu_supports("avx2")) { vnni256 = __builtin_cpu_supports("avxvnni"); f.simd = vnni256 ? 2 : 1;
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vnni")) f.simd = 3; }   // yvvyee: AVX-512 (2026-11-04)
#endif
  if (getenv("NNI_SIMD")) f.simd = std::min(f.simd, atoi(getenv("NNI_SIMD")));   // yvvyee: 시험용 (0/1 로 낮춰 경로별 비트 일치 확인)
  if (f.simd == 2 && !vnni256) f.simd = 1;
  if (!getenv("NNI_REF")) m_fast = F;   // yvvyee: NNI_REF 가 있으면 기준 구현만 쓴다 (시험용)
  if (getenv("NNI_VERBOSE")) fprintf(stderr, "NNInterp: fast=%d fc=%d simd=%d\n", m_fast ? 1 : 0, f.fc, f.simd);   // yvvyee: 시험용 (가속 경로 사용 여부 확인)
}

void NNInterp::predict(const BlockIn &in, BlockOut &out) const { predict(in, out, false); }

void NNInterp::predict(const BlockIn &in, BlockOut &out, bool linearOnly) const
{
  if (!m_fast)
  {
    predictRef(in, out);
    if (linearOnly)   // yvvyee: reference path computes ylin as well; return its tile area as y
      for (int yy = 0; yy < in.th; yy++) for (int xx = 0; xx < in.tw; xx++) out.y[yy * in.tw + xx] = out.ylin[(yy + 2) * (in.tw + 4) + xx + 2];
    return;
  }
  const Fast &F = *m_fast; Work &w = work();
  const int nb = 4, K = F.K;
  const int st = in.st, qp = in.qp, fx = std::max(0, in.fx), fy = std::max(0, in.fy);
  const int imv = in.imv, merge = in.merge, dpoc = in.dpoc, w4 = in.w4, h4 = in.h4;
  const int tw = in.tw, th = in.th, Hh = th + 16, Ww = tw + 16, HW = Hh * Ww, Ly = th + 4, Lx = tw + 4, Cy = th + 12, Cx = tw + 12;
  if (tw < 4 || th < 4 || tw > 16 || th > 16 || (tw & 3) || (th & 3) || in.stride < Ww || in.stride * (Hh - 1) + Ww > 1024 || K > 64)
  {
    fprintf(stderr, "NNInterp: bad tile tw=%d th=%d stride=%d K=%d\n", tw, th, in.stride, K); exit(1);
  }
  int64_t *x = w.x; int *cm = w.cm;
  for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) { x[yy * Ww + xx] = (int64_t) in.win[yy * in.stride + xx] << PIXF; cm[yy * Ww + xx] = in.map[yy * in.stride + xx]; }
  int64_t sd[16] = { 0 }; sd[0] = fdiv_round((i128) qp * 16384, 63); sd[1 + clampi(imv, 0, 3)] = 16384;
  { const bool integ = (fx == 0 && fy == 0), half = !integ && (fx % 8 == 0) && (fy % 8 == 0), quar = !integ && !half && (fx % 4 == 0) && (fy % 4 == 0);
    sd[5 + (integ ? 0 : half ? 1 : quar ? 2 : 3)] = 16384; }
  sd[9] = merge ? 16384 : 0; sd[10] = clamp16(fdiv_round((i128) std::abs(dpoc) * 16384, 16)); sd[11] = F.tabB->v[clampi(w4, 1, 32) - 1]; sd[12] = F.tabB->v[clampi(h4, 1, 32) - 1]; sd[13 + clampi(st, 0, 2)] = 16384;
  int64_t (*cf)[MAXHW] = w.cf; codemapFeatsF(cm, Hh, Ww, qp, cf);
  // ---- L1
  sepFirF(x, Hh, Ww, *F.g06, w.g0, w.tmp, 0, 0, 0, 32767); sepFirF(x, Hh, Ww, *F.g12, w.g1, w.tmp, 0, 0, 0, 32767); sepFirF(x, Hh, Ww, *F.g24, w.g2, w.tmp, 0, 0, 0, 32767);
  int64_t *B[4] = { w.g2, w.B[1], w.B[2], w.B[3] };
  for (int i = 0; i < HW; i++) { B[1][i] = w.g1[i] - w.g2[i]; B[2][i] = w.g0[i] - w.g1[i]; B[3][i] = x[i] - w.g0[i]; }
  // ---- L2
  for (int b = 0; b < 4; b++) { for (int i = 0; i < HW; i++) w.sq[i] = energyQF(B[b][i], B[b][i], F.eSq[b], 0); sepFirF(w.sq, Hh, Ww, *F.g10, w.phi[b], w.tmp, F.eSq[b], F.ePhi[b], F.ePhi[b], 2147483647LL); }
  // ---- L3
  {
    const int64_t *xs[28]; int sIn[28]; bool isC[28]; int64_t sdc[16][1];
    for (int c = 0; c < 16; c++) { sdc[c][0] = sd[c]; xs[c] = sdc[c]; sIn[c] = 14; isC[c] = true; }
    for (int c = 0; c < 12; c++) { xs[16 + c] = cf[c]; sIn[16 + c] = 14; isC[16 + c] = false; }
    conv1x1x2F(F.simd, F.n0, F.n2, xs, sIn, isC, nullptr, HW, w.h2);
  }
  const int ePhin = F.ePhin, s2 = F.n2.oexp; const int64_t INV30 = INV30_Q;
  for (int b = 0; b < nb; b++) for (int p = 0; p < HW; p++)
  {
    const int64_t lphi = clamp16(rshr(((i128) w.h2[b][p] << 8) - ((i128) 7680 << s2), s2));
    int64_t t; int n; expQF(lphi, t, n); w.phin[b][p] = clamp32(rshr(t, 15 - n - ePhin));
    const int Kc = std::max(ePhin, 40); i128 num = ((i128) w.phin[b][p] << (Kc - ePhin)) + ((i128) 1 << (Kc - 40));
    const int64_t ln10 = lnQ10F(num, Kc); w.lpn[b][p] = clamp16(rshr((i128) ln10 * INV30, 12));
  }
  // ---- L4
  int64_t *xd = w.xd;
  for (int p = 0; p < HW; p++)
  {
    i128 acc = 0;
    for (int b = 0; b < nb; b++)
    {
      const int Kc = std::max(std::max(F.ePhi[b], ePhin), 40);
      i128 den = ((i128) w.phi[b][p] << (Kc - F.ePhi[b])) + ((i128) w.phin[b][p] << (Kc - ePhin)) + ((i128) 1 << (Kc - 40));
      int64_t r15; int e; recipQF(den, Kc, r15, e);
      const int64_t G15 = clamp32(rshr((i128) w.phi[b][p] * r15, F.ePhi[b] + e));
      acc += (i128) G15 * B[b][p];
    }
    xd[p] = rshr(acc, 15);
  }
  // ---- L4b (원식)
  { const NNInterp::Arr &k7 = *F.k7; const int64_t gate = F.gate; const bool ax = (imv == 3 && fx == 8), ay = (imv == 3 && fy == 8); int64_t *l = w.lxy;
    if (ax) { for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) { i128 acc = 0; for (int k = 0; k < 7; k++) acc += (i128) xd[yy * Ww + clampi(xx + k - 3, 0, Ww - 1)] * k7.v[k]; l[yy * Ww + xx] = rshr(acc, FIRF); }
      for (int p = 0; p < HW; p++) xd[p] = rshr(((i128) xd[p] << 15) + (i128) gate * (l[p] - xd[p]), 15); }
    if (ay) { for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) { i128 acc = 0; for (int k = 0; k < 7; k++) acc += (i128) xd[clampi(yy + k - 3, 0, Hh - 1) * Ww + xx] * k7.v[k]; l[yy * Ww + xx] = rshr(acc, FIRF); }
      for (int p = 0; p < HW; p++) xd[p] = rshr(((i128) xd[p] << 15) + (i128) gate * (l[p] - xd[p]), 15); } }
  for (int i = 0; i < HW; i++) out.xd[i] = xd[i];
  // ---- L6 쿼리 입력 (lst 와 1x1 은 주의 영역 화소만)
  const int eSt0 = F.eSt0, eSt = F.eSt, NL = Ly * Lx; int *pix = w.pix;
  for (int t = 0; t < NL; t++) pix[t] = (6 + t / Lx) * Ww + 6 + t % Lx;
  for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) { w.gx[yy * Ww + xx] = xd[yy * Ww + clampi(xx + 1, 0, Ww - 1)] - xd[yy * Ww + clampi(xx - 1, 0, Ww - 1)]; w.gy[yy * Ww + xx] = xd[clampi(yy + 1, 0, Hh - 1) * Ww + xx] - xd[clampi(yy - 1, 0, Hh - 1) * Ww + xx]; }
  for (int c = 0; c < 3; c++)
  {
    const int64_t *ga = c == 0 ? w.gx : (c == 1 ? w.gx : w.gy), *gb = c == 0 ? w.gx : w.gy;
    for (int p = 0; p < HW; p++) w.sq[p] = energyQF(ga[p], gb[p], eSt0, 2);
    sepFirF(w.sq, Hh, Ww, *F.g10, w.s1v, w.tmp, eSt0, eSt, eSt, 2147483647LL);
    for (int t = 0; t < NL; t++)
    {
      const int p = pix[t]; const int64_t v = w.s1v[p] * 1023; const int sg = v < 0 ? -1 : (v > 0 ? 1 : 0);
      i128 num = (i128) (v < 0 ? -v : v) + ((i128) 1 << eSt); const int64_t l10 = lnQ10F(num, eSt);
      w.lst[c][p] = clamp16(rshr((i128) sg * l10, 2));
    }
  }
  {
    const int64_t *xs[24]; int sIn[24]; bool isC[24]; int64_t sdc[16][1];
    for (int c = 0; c < 16; c++) { sdc[c][0] = sd[c]; xs[c] = sdc[c]; sIn[c] = 14; isC[c] = true; }
    for (int c = 0; c < 3; c++) { xs[16 + c] = w.lst[c]; sIn[16 + c] = 8; isC[16 + c] = false; }
    xs[19] = w.lpn[3]; sIn[19] = 14; isC[19] = false; for (int c = 0; c < 4; c++) { xs[20 + c] = cf[c]; sIn[20 + c] = 14; isC[20 + c] = false; }
    conv1x1x2F(F.simd, F.q0, F.q2, xs, sIn, isC, pix, NL, w.q);
  }
  const int sq = F.q2.oexp;
  // ---- L5 후보 이동: 가로 패스는 열 2..2+Cx (L7 정렬 창과 같은 범위) 로 이동별 한 번, 후보는 열 오프셋 4
  const NNInterp::Arr &K12 = *F.k12; const bool i64 = maxAbs(xd, HW) < (1LL << 40);
  int hsx[26], nh = 0, hid[64];
  auto hget = [&](int sx) { for (int i = 0; i < nh; i++) if (hsx[i] == sx) return i; hsx[nh] = sx; hPassF(xd, Hh, Ww, sx, 2, Cx, K12, i64, w.Xh[nh]); return nh++; };
  const int hA0 = hget(fx);
  for (int c = 0; c < K; c++) hid[c] = hget(fx + (int) F.cdx->v[c]);
  int64_t hmax[26]; for (int i = 0; i < nh; i++) hmax[i] = maxAbs(w.Xh[i], Hh * Cx);
  for (int c = 0; c < K; c++)
  {
    const int64_t *Xh = w.Xh[hid[c]];
    if (F.simd >= 1 && hmax[hid[c]] < (1LL << 31) && (Lx & 3) == 0) vPassAvx2(Xh, Hh, Cx, 4, fy + (int) F.cdy->v[c], 6, Ly, Lx, K12, w.Vc[c]);
    else vPassF(Xh, Hh, Cx, 4, fy + (int) F.cdy->v[c], 6, Ly, Lx, K12, i64 && hmax[hid[c]] < (1LL << 40), w.Vc[c]);
  }
  // ---- L6 주의
  const int64_t *phiM = F.phiM->v.data(); const int lsh = sq + F.ephi - 8;
  for (int t = 0; t < NL; t++)
  {
    const int p = pix[t]; int64_t lg[64]; int64_t mx = INT64_MIN;
    for (int c = 0; c < K; c++)
    {
      if (lsh >= 1 && lsh <= 62) { int64_t acc = 0; for (int j = 0; j < 5; j++) acc += w.q[j][p] * phiM[c * 5 + j]; lg[c] = rshr(acc, lsh); }   // yvvyee: |q| <= 32767, |phi| <= 2^15 : 합 < 2^33
      else { i128 acc = 0; for (int j = 0; j < 5; j++) acc += (i128) w.q[j][p] * phiM[c * 5 + j]; lg[c] = rshr(acc, lsh); }
      mx = std::max(mx, lg[c]);
    }
    int64_t e45[64]; i128 sum = 0;
    for (int c = 0; c < K; c++) { int64_t tt; int n; expQF(lg[c] - mx, tt, n); e45[c] = rshr(tt, 15 - n - 45); sum += e45[c]; }
    int64_t r15; int e; recipQF(sum, 45, r15, e); i128 acc = 0;
    for (int c = 0; c < K; c++) { const int64_t a15 = clamp32(rshr((i128) e45[c] * r15, 45 + e)); acc += (i128) a15 * w.Vc[c][t]; }
    w.ylin[t] = rshr(acc, 15);
  }
  for (int i = 0; i < NL; i++) out.ylin[i] = w.ylin[i];
  if (linearOnly)
  {
    // yvvyee: proxy (encoder screening only). Same indexing as the L8 output line: y = ylin[(y+2)*Lx + x+2] without the rate-gated correction.
    for (int yy = 0; yy < th; yy++) for (int xx = 0; xx < tw; xx++) out.y[yy * tw + xx] = w.ylin[(yy + 2) * Lx + xx + 2];
    return;
  }
  // ---- L7
  const int eIn = F.eIn;
  { const int64_t *Xh = w.Xh[hA0];
    if (F.simd >= 1 && hmax[hA0] < (1LL << 31) && (Cx & 3) == 0) vPassAvx2(Xh, Hh, Cx, 0, fy, 2, Cy, Cx, K12, w.Aw);
    else vPassF(Xh, Hh, Cx, 0, fy, 2, Cy, Cx, K12, i64 && hmax[hA0] < (1LL << 40), w.Aw); }
  {
    int16_t *a0 = w.act; const int cfi[6] = { 0, 1, 2, 3, 8, 9 };
    for (int yy = 0; yy < Cy; yy++) for (int xx = 0; xx < Cx; xx++)
    {
      const int i = yy * Cx + xx, s = (yy + 2) * Ww + xx + 2; int16_t *d = a0 + i * 12;
      d[0] = (int16_t) clamp16(rdivPow2(((i128) w.Aw[i]) << eIn, 14));   // fdiv_round(., PIX_DEN = 2^14)
      for (int b = 0; b < 4; b++) d[1 + b] = (int16_t) w.lpn[b][s];
      for (int k = 0; k < 6; k++) d[5 + k] = (int16_t) cf[cfi[k]][s];
      d[11] = 0;
    }
    const int Wp = Cx + 2;
    (F.simd >= 3 ? packPadZ : packPad)(a0, Cy, Cx, 12, F.c3[0].NP, w.XH, w.XL);   // yvvyee: AVX-512 (2026-11-04)
    conv3x3F(F.simd, F.c3[0], w.XH, w.XL, Wp, 0, 0, Cy, Cx, w.act, nullptr, Cx);
    for (int l = 0; l < 4; l++)
    {
      filtActF(F.simd, F.fa[l], w.act, Cy, Cx, w.fa, w, F.fc);
      (F.simd >= 3 ? packPadZ : packPad)(w.fa, Cy, Cx, F.fc, F.c3[l + 1].NP, w.XH, w.XL);   // yvvyee: AVX-512 (2026-11-04)
      if (l < 3) conv3x3F(F.simd, F.c3[l + 1], w.XH, w.XL, Wp, 0, 0, Cy, Cx, w.act, nullptr, Cx);
      else conv3x3F(F.simd, F.c3[4], w.XH, w.XL, Wp, 6, 6, th, tw, nullptr, w.c4, Cx);   // yvvyee: 마지막 층은 L8 이 읽는 타일 영역만
    }
  }
  const int sc4 = F.sc4;
  // ---- L8 (원식, 2의 거듭제곱 분모 나눗셈만 시프트로)
  const NNInterp::Arr &D = *F.D; const int eD = F.eD; const int64_t thq = F.rth->v[clampi(qp, 0, 63)];
  int eth = -60; while (((i128) PIX_DEN << (eth + 1 + 60)) <= ((i128) thq << 60)) eth++;
  int64_t r15th;
  { i128 numm = (i128) thq * 1024, denm = (i128) PIX_DEN; if (eth >= 0) denm <<= eth; else numm <<= (-eth);
    const int m10 = (int) fdiv_round(numm, denm); r15th = LUT_RECIP->at(m10 - 1024); }
  const int kdd = 29 + (eth >= 0 ? eth : 0);   // dd = PIX_DEN << 15 << max(eth,0) = 2^kdd
  for (int by = 0; by < th; by += 4)
    for (int bx = 0; bx < tw; bx += 4)
    {
      int64_t r[16], rq[16], c[16], cc[16]; for (int yy = 0; yy < 4; yy++) for (int xx = 0; xx < 4; xx++) r[yy * 4 + xx] = w.c4[(by + yy + 6) * Cx + bx + xx + 6];
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { i128 acc = 0; for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) acc += (i128) D.v[i * 4 + k] * r[k * 4 + l] * D.v[j * 4 + l]; c[i * 4 + j] = rshr(acc * PIX_DEN, 2 * eD + sc4); }
      for (int i = 0; i < 16; i++)
      {
        const int64_t ac = c[i] < 0 ? -c[i] : c[i];
        i128 nn = (i128) 4 * (ac - thq) * r15th * 256; if (eth < 0) nn <<= (-eth);
        const int64_t arg = clamp32(rdivPow2(nn, kdd)); const int64_t s15 = sigQ15(arg);
        cc[i] = rshr((i128) c[i] * s15, 15);
      }
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { i128 acc = 0; for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) acc += (i128) D.v[k * 4 + i] * cc[k * 4 + l] * D.v[l * 4 + j]; rq[i * 4 + j] = rshr(acc, 2 * eD); }
      for (int yy = 0; yy < 4; yy++) for (int xx = 0; xx < 4; xx++) out.y[(by + yy) * tw + bx + xx] = w.ylin[(by + yy + 2) * Lx + bx + xx + 2] + rq[yy * 4 + xx];
    }
}
