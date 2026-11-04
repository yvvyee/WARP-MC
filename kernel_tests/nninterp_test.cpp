// yvvyee: QKV-MC 가설별 보간(L1~L8, L9 없음) 고정소수 C++ 커널 + 비트 일치 시험기.
// 모사기 qkv_fixed.py(float64, 양자화 전부 켬)와 같은 산술. 반올림 floor(x+0.5) = (num + half) >> shift (산술 시프트, i128).
// 화소 Q10.4 (정규화 1.0 = 1023*16 = 16368), FIR Q1.14, 활성·에너지는 보정 지수 2^e 의 정수, 조회표는 내보내기 값.
// 사용: nninterp_test <export_dir> <dump.dmp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

typedef __int128 i128;
typedef std::vector<int64_t> V;
struct Arr { std::string dtype; std::vector<int64_t> shape; V v; };
static std::map<std::string, Arr> g;
static const Arr &A(const std::string &n) { auto it = g.find(n); if (it == g.end()) { fprintf(stderr, "missing %s\n", n.c_str()); exit(1); } return it->second; }
static int64_t S(const std::string &n, int i = 0) { return A(n).v[i]; }

static void loadExport(const std::string &dir)
{
  FILE *mf = fopen((dir + "/manifest.txt").c_str(), "r"); if (!mf) { fprintf(stderr, "no manifest\n"); exit(1); }
  char name[256], dtype[32], fn[256]; int nd;
  while (fscanf(mf, "%255s %31s %d", name, dtype, &nd) == 3)
  {
    Arr a; a.dtype = dtype; int64_t cnt = 1;
    for (int i = 0; i < nd; i++) { long long d; if (fscanf(mf, "%lld", &d) != 1) exit(1); a.shape.push_back(d); cnt *= d; }
    if (fscanf(mf, "%255s", fn) != 1) exit(1);
    FILE *f = fopen((dir + "/" + fn).c_str(), "rb"); a.v.resize(cnt);
    if (a.dtype == "int16") { std::vector<int16_t> t(cnt); if (fread(t.data(), 2, cnt, f) != (size_t) cnt) exit(1); for (int64_t i = 0; i < cnt; i++) a.v[i] = t[i]; }
    else { std::vector<int32_t> t(cnt); if (fread(t.data(), 4, cnt, f) != (size_t) cnt) exit(1); for (int64_t i = 0; i < cnt; i++) a.v[i] = t[i]; }
    fclose(f); g[name] = a;
  }
  fclose(mf);
}

// ---------------------------------------------------------------- 산술
static const int WS = 20, PIXF = 4, FIRF = 14;
static const int64_t PIX_DEN = 1024LL << PIXF;   // 정규화 1.0 = 1024 화소 (2진)
static inline int64_t rshr(i128 x, int sh)   // floor(x / 2^sh + 0.5); sh<0 은 좌시프트
{
  if (sh <= 0) return (int64_t) (x << (-sh));
  if (sh >= 120) return 0;
  i128 h = (i128) 1 << (sh - 1); return (int64_t) ((x + h) >> sh);
}
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
static const Arr *LUT_EXP2, *LUT_LOG2, *LUT_RECIP, *LUT_SIG; static int64_t LOG2E_Q, LN2_Q, INV30_Q;

// e^x, x = x8/256 (Q8.8) -> 결과 = t * 2^(n-15), (t = 2^f 표 Q0.15). 반환 (t, n)
static inline void expQ(int64_t x8, int64_t &t, int &n)
{
  int64_t t8 = rshr((i128) x8 * LOG2E_Q, 16); t8 = std::max<int64_t>(-86 * 256, std::min<int64_t>(86 * 256, t8));
  n = (int) fdiv_floor(t8, 256); const int f = (int) (t8 - 256LL * n); t = LUT_EXP2->v[f];
}
// ln(값), 값 = num * 2^-sc : log2 = e + T[m]/1024 (Q10) ; ln(Q10) = rshr((e*1024+T)*LN2_Q, 16)
static inline int64_t lnQ10(i128 num, int sc)
{
  int e, m10; normalize(num, sc, e, m10); const int64_t l2 = (int64_t) e * 1024 + LUT_LOG2->v[m10 - 1024];
  return rshr((i128) l2 * LN2_Q, 16);
}
// 1/값 = r15 * 2^-15 * 2^-e
static inline void recipQ(i128 num, int sc, int64_t &r15, int &e) { int m10; normalize(num, sc, e, m10); r15 = LUT_RECIP->v[m10 - 1024]; }
static inline int64_t sigQ15(int64_t x8) { x8 = std::max<int64_t>(-4096, std::min<int64_t>(4096, x8)); return LUT_SIG->v[x8 + 4096]; }

// 분리형 FIR (복제 패딩) : 입력 격자 2^sIn, 출력 격자 2^sOut, 탭 Q1.14, 각 패스 후 반올림
static void sepFirRep(const V &x, int H, int W, const Arr &h, V &y, int sIn, int sMid, int sOut)
{
  const int T = (int) h.v.size(), r = (T - 1) / 2; V t(H * W); y.assign(H * W, 0);
  for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) { i128 acc = 0; for (int k = 0; k < T; k++) acc += (i128) x[yy * W + clampi(xx + k - r, 0, W - 1)] * h.v[k]; t[yy * W + xx] = rshr(acc, FIRF + sIn - sMid); }
  for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) { i128 acc = 0; for (int k = 0; k < T; k++) acc += (i128) t[clampi(yy + k - r, 0, H - 1) * W + xx] * h.v[k]; y[yy * W + xx] = rshr(acc, FIRF + sMid - sOut); }
}
// 에너지: (a*b) 정규화 (a,b 는 Q10.4, 추가 분모 extraDen) 를 2^e 격자로
static inline int64_t energyQ(int64_t a, int64_t b, int e, int64_t extraDen)
{
  i128 num = (i128) a * b; if (e >= 0) num <<= e; i128 den = (i128) PIX_DEN * PIX_DEN * extraDen;
  if (e < 0) den <<= (-e);
  return clamp32(fdiv_round(num, den));
}

// 창 sinc 이동: 입력 X (H x W, Q10.4), 이동 (sx16, sy16) 1/16 단위, 영역 [r0, r0+n) -> n x n
static void shiftCrop(const V &X, int H, int W, int sx16, int sy16, int r0, int n, V &out)
{
  const Arr &K = A("fir_kaiser16x12"); V Xh(H * n);
  for (int t = 0; t < n; t++)
  {
    const int p16 = (r0 + t) * 16 + sx16; const int i0 = (int) fdiv_floor(p16, 16), f = p16 - 16 * i0;
    for (int yy = 0; yy < H; yy++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) X[yy * W + clampi(i0 + k - 5, 0, W - 1)] * K.v[f * 12 + k]; Xh[yy * n + t] = rshr(acc, FIRF); }
  }
  out.assign(n * n, 0);
  for (int t = 0; t < n; t++)
  {
    const int p16 = (r0 + t) * 16 + sy16; const int i0 = (int) fdiv_floor(p16, 16), f = p16 - 16 * i0;
    for (int xx = 0; xx < n; xx++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) Xh[clampi(i0 + k - 5, 0, H - 1) * n + xx] * K.v[f * 12 + k]; out[t * n + xx] = rshr(acc, FIRF); }
  }
}

// 1x1 합성곱 (채널별 입력 스케일 지수 sIn[c]) -> 출력 격자 2^sOut, 편향 Q1.14
static void conv1x1(const std::vector<V> &x, const std::vector<int> &sIn, int HW, const std::string &nm, std::vector<V> &y, int &sOut, bool relu)
{
  const Arr &w = A(nm + "_w"), &b = A(nm + "_b_q14"); const int we = (int) S(nm + "_wexp"); sOut = (int) S(nm + "_oexp");
  const int Co = (int) w.shape[0], Ci = (int) w.shape[1]; y.assign(Co, V(HW));
  const int SM = 40;   // 공통 스케일 2^SM 로 누산 (입력 지수 최대 < 40 가정)
  for (int o = 0; o < Co; o++) for (int p = 0; p < HW; p++)
  {
    i128 acc = 0;
    for (int c = 0; c < Ci; c++) acc += ((i128) w.v[o * Ci + c] * x[c][p]) << (SM - sIn[c]);           // 스케일 2^(we+SM)
    acc += ((i128) b.v[o]) << (we + SM - 14);
    int64_t v = clamp16(rshr(acc, we + SM - sOut)); if (relu && v < 0) v = 0; y[o][p] = v;
  }
}
// 3x3 합성곱 (반사 패딩 1), 채널별 입력 스케일. LeakyReLU 는 입력에 (분자 51, 분모 256) 로 전달됨: xNum[c][p] = x*256 (양) / x*51 (음)
static void conv3x3(const std::vector<V> &xNum, const std::vector<int> &sIn, int Hh, int Ww, const std::string &nm, std::vector<V> &y, int &sOut)
{
  const Arr &w = A(nm + "_w"), &b = A(nm + "_b_q14"); const int we = (int) S(nm + "_wexp"); sOut = (int) S(nm + "_oexp");
  const int Co = (int) w.shape[0], Ci = (int) w.shape[1]; y.assign(Co, V(Hh * Ww)); const int SM = 40;
  auto refl = [](int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); };
  for (int o = 0; o < Co; o++) for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++)
  {
    i128 acc = 0;
    for (int c = 0; c < Ci; c++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
      acc += ((i128) w.v[((o * Ci + c) * 3 + ky) * 3 + kx] * xNum[c][refl(yy + ky - 1, Hh) * Ww + refl(xx + kx - 1, Ww)]) << (SM - 8 - sIn[c]);   // 분모 256 반영
    acc += ((i128) b.v[o]) << (we + SM - 14);
    y[o][yy * Ww + xx] = clamp16(rshr(acc, we + SM - sOut));
  }
}
// FiltAct: 2배 업샘플(0삽입) -> sep(hu, 키 h/v) -> lrelu -> sep(hd, 키 h/v) -> [::2]. 입력 격자 sIn, 출력 격자 sOut(=ev)
static void filtAct(const V &x, int Hh, int Ww, int sIn, const std::string &nm, V &y, int &sOut)
{
  const Arr &hu = A("fir_filtact_hu_half"), &hd = A("fir_filtact_hd"); const int eh = (int) S(nm + "_oexp", 0), ev = (int) S(nm + "_oexp", 1);
  const int H2 = 2 * Hh, W2 = 2 * Ww; V u(H2 * W2, 0); for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) u[(2 * yy) * W2 + 2 * xx] = x[yy * Ww + xx];
  auto refl = [](int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); };
  // sep(hu): 탭 h14*2 (hu_half 표 x2), 반사 패딩 (6 좌, 5 우): 인덱스 k-6
  V t(H2 * W2), a(H2 * W2);
  for (int yy = 0; yy < H2; yy++) for (int xx = 0; xx < W2; xx++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) u[yy * W2 + refl(xx + k - 6, W2)] * (2 * hu.v[k]); t[yy * W2 + xx] = clamp16(rshr(acc, FIRF + sIn - eh)); }
  for (int yy = 0; yy < H2; yy++) for (int xx = 0; xx < W2; xx++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) t[refl(yy + k - 6, H2) * W2 + xx] * (2 * hu.v[k]); a[yy * W2 + xx] = clamp16(rshr(acc, FIRF + eh - ev)); }
  // lrelu (분자 51 / 분모 256), 격자 ev
  V num(H2 * W2); for (int i = 0; i < H2 * W2; i++) num[i] = a[i] >= 0 ? a[i] * 256 : a[i] * 51;
  // sep(hd): 입력 num/256 at 2^ev
  for (int yy = 0; yy < H2; yy++) for (int xx = 0; xx < W2; xx++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) num[yy * W2 + refl(xx + k - 6, W2)] * hd.v[k]; t[yy * W2 + xx] = clamp16(rshr(acc, FIRF + 8 + ev - eh)); }
  for (int yy = 0; yy < H2; yy++) for (int xx = 0; xx < W2; xx++) { i128 acc = 0; for (int k = 0; k < 12; k++) acc += (i128) t[refl(yy + k - 6, H2) * W2 + xx] * hd.v[k]; a[yy * W2 + xx] = clamp16(rshr(acc, FIRF + eh - ev)); }
  y.assign(Hh * Ww, 0); for (int yy = 0; yy < Hh; yy++) for (int xx = 0; xx < Ww; xx++) y[yy * Ww + xx] = a[(2 * yy) * W2 + 2 * xx]; sOut = ev;
}

// 지도 특징: 경계 플래그 스캔 (codemap_feats 와 동일), 결과 Q1.14 (12채널)
static void codemapFeats(const std::vector<int> &cm, int qp, std::vector<V> &cf)
{
  cf.assign(12, V(WS * WS, 0)); const int n = WS;
  auto scan = [&](bool horiz) {
    for (int a = 0; a < n; a++)   // a: 고정 축 (행 또는 열)
    {
      int last = -1; int64_t vb = 0;
      for (int i = 0; i < n; i++)
      {
        const int p = horiz ? a * n + i : i * n + a; const int flag = horiz ? (cm[p] & 1) : ((cm[p] >> 1) & 1); const int bs = horiz ? ((cm[p] >> 2) & 3) : ((cm[p] >> 4) & 3);
        if (flag) { last = i; vb = bs; }
        const int64_t d = last >= 0 ? i - last : n; const int64_t dq = std::min<int64_t>(d, 8) * 2048;   // d/8 * 16384
        if (horiz) { cf[0][p] = dq; cf[4][p] = last >= 0 ? vb * 8192 : 0; } else { cf[2][p] = dq; cf[6][p] = last >= 0 ? vb * 8192 : 0; }
      }
      int nxt = n + 1; int64_t vf = 0;
      for (int i = n - 1; i >= 0; i--)
      {
        const int p = horiz ? a * n + i : i * n + a;
        const int64_t d = (nxt <= n) ? (nxt - 1 - i) : n; const int64_t dq = std::min<int64_t>(d, 8) * 2048;
        if (horiz) { cf[1][p] = dq; cf[5][p] = (nxt <= n) ? vf * 8192 : 0; } else { cf[3][p] = dq; cf[7][p] = (nxt <= n) ? vf * 8192 : 0; }
        const int flag = horiz ? (cm[p] & 1) : ((cm[p] >> 1) & 1); const int bs = horiz ? ((cm[p] >> 2) & 3) : ((cm[p] >> 4) & 3);
        if (flag) { nxt = i; vf = bs; }   // e >= x+1 중 최소: 현재 위치 플래그는 다음(작은) 인덱스에서 보임
      }
    }
  };
  scan(true); scan(false);
  for (int p = 0; p < n * n; p++)
  {
    const int valid = (cm[p] >> 14) & 1, qpr = (cm[p] >> 8) & 63;
    cf[8][p] = ((cm[p] >> 6) & 1) * 16384; cf[9][p] = ((cm[p] >> 7) & 1) * 16384;
    cf[10][p] = valid ? clamp16(fdiv_round((i128) (qpr - qp) * 16384, 12)) : 0; cf[11][p] = valid * 16384;
  }
}

struct Out { V xd, ylin, y; };

static void hypothesis(const std::vector<int16_t> &rec, Out &o)
{
  const int nb = 4; const int K = (int) A("cand_dx_q4").shape[0];
  const int st = rec[0], qp = rec[1], fx = std::max<int>(0, rec[5]), fy = std::max<int>(0, rec[6]);
  const int imv = rec[1130], merge = rec[1131], dpoc = rec[1133], w4 = (rec[1135] >> 8) & 255, h4 = rec[1135] & 255;
  // ---- 입력 창 Q10.4, 지도
  V x(WS * WS); std::vector<int> cm(WS * WS); for (int i = 0; i < WS * WS; i++) { x[i] = (int64_t) rec[330 + i] << PIXF; cm[i] = rec[1136 + i]; }
  // ---- 부가 입력 sd (16, Q1.14)
  V sd(16, 0); sd[0] = fdiv_round((i128) qp * 16384, 63); sd[1 + clampi(imv, 0, 3)] = 16384;
  { const bool integ = (fx == 0 && fy == 0), half = !integ && (fx % 8 == 0) && (fy % 8 == 0), quar = !integ && !half && (fx % 4 == 0) && (fy % 4 == 0);
    sd[5 + (integ ? 0 : half ? 1 : quar ? 2 : 3)] = 16384; }
  sd[9] = merge ? 16384 : 0; sd[10] = clamp16(fdiv_round((i128) std::abs(dpoc) * 16384, 16)); sd[11] = S("tab_log2bsz_q14", clampi(w4, 1, 32) - 1); sd[12] = S("tab_log2bsz_q14", clampi(h4, 1, 32) - 1); sd[13 + clampi(st, 0, 2)] = 16384;
  std::vector<V> cf; codemapFeats(cm, qp, cf);
  // ---- L1 대역 분해 (Q10.4)
  V g0, g1, g2; sepFirRep(x, WS, WS, A("fir_gauss06"), g0, 0, 0, 0); sepFirRep(x, WS, WS, A("fir_gauss12"), g1, 0, 0, 0); sepFirRep(x, WS, WS, A("fir_gauss24"), g2, 0, 0, 0);
  V B[4]; B[0] = g2; for (int b = 1; b < 4; b++) B[b].resize(WS * WS);
  for (int i = 0; i < WS * WS; i++) { B[1][i] = g1[i] - g2[i]; B[2][i] = g0[i] - g1[i]; B[3][i] = x[i] - g0[i]; }
  // ---- L2 phi_s (에너지 격자 phi_b)
  int eSq[4], ePhi[4]; for (int b = 0; b < 4; b++) { eSq[b] = (int) S("energy_sq" + std::to_string(b) + "_exp"); ePhi[b] = (int) S("energy_phi" + std::to_string(b) + "_exp"); }
  V phi[4];
  for (int b = 0; b < 4; b++) { V sq(WS * WS); for (int i = 0; i < WS * WS; i++) sq[i] = energyQ(B[b][i], B[b][i], eSq[b], 1); sepFirRep(sq, WS, WS, A("fir_gauss10"), phi[b], eSq[b], ePhi[b], ePhi[b]); }
  // ---- L3 잡음 쿼리: [sd(16), cf(12)] -> 1x1 (relu) -> 1x1 -> lphi (Q8.8) -> phi_n (에너지 phin)
  std::vector<V> xin(28, V(WS * WS)); std::vector<int> sIn(28, 14);
  for (int c = 0; c < 16; c++) std::fill(xin[c].begin(), xin[c].end(), sd[c]);
  for (int c = 0; c < 12; c++) xin[16 + c] = cf[c];
  std::vector<V> h1, h2; int s1, s2; conv1x1(xin, sIn, WS * WS, "conv_noise_0", h1, s1, true); conv1x1(h1, std::vector<int>(32, s1), WS * WS, "conv_noise_2", h2, s2, false);
  const int ePhin = (int) S("energy_phin_exp");
  std::vector<V> lphi(nb, V(WS * WS)), phin(nb, V(WS * WS)), lpn(nb, V(WS * WS));
  const int64_t INV30 = INV30_Q;
  for (int b = 0; b < nb; b++) for (int p = 0; p < WS * WS; p++)
  {
    lphi[b][p] = clamp16(rshr(((i128) h2[b][p] << 8) - ((i128) 7680 << s2), s2));                              // Q8.8
    int64_t t; int n; expQ(lphi[b][p], t, n); phin[b][p] = clamp32(rshr(t, 15 - n - ePhin));                   // 에너지 격자
    // lpn = Q1.14( ln(phi_n + 2^-40) * INV30 ) ; 값 = phin/2^ePhin + 2^-40
    const int Kc = std::max(ePhin, 40); i128 num = ((i128) phin[b][p] << (Kc - ePhin)) + ((i128) 1 << (Kc - 40));
    const int64_t ln10 = lnQ10(num, Kc); lpn[b][p] = clamp16(rshr((i128) ln10 * INV30, 12));
  }
  // ---- L4 Wiener: G15 = rshr(phi_s * r15, ePhi + e) ; xd = rshr(sum G15*B, 15)
  V xd(WS * WS);
  for (int p = 0; p < WS * WS; p++)
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
  { const Arr &k7 = A("alt_kernel7"); const int64_t gate = S("alt_gate_q15"); const bool ax = (imv == 3 && fx == 8), ay = (imv == 3 && fy == 8);
    if (ax) { V lx(WS * WS); for (int yy = 0; yy < WS; yy++) for (int xx = 0; xx < WS; xx++) { i128 acc = 0; for (int k = 0; k < 7; k++) acc += (i128) xd[yy * WS + clampi(xx + k - 3, 0, WS - 1)] * k7.v[k]; lx[yy * WS + xx] = rshr(acc, FIRF); }
      for (int p = 0; p < WS * WS; p++) xd[p] = rshr(((i128) xd[p] << 15) + (i128) gate * (lx[p] - xd[p]), 15); }
    if (ay) { V ly(WS * WS); for (int yy = 0; yy < WS; yy++) for (int xx = 0; xx < WS; xx++) { i128 acc = 0; for (int k = 0; k < 7; k++) acc += (i128) xd[clampi(yy + k - 3, 0, WS - 1) * WS + xx] * k7.v[k]; ly[yy * WS + xx] = rshr(acc, FIRF); }
      for (int p = 0; p < WS * WS; p++) xd[p] = rshr(((i128) xd[p] << 15) + (i128) gate * (ly[p] - xd[p]), 15); } }
  o.xd = xd;
  // ---- L6 쿼리 입력: 구조 텐서 (gx, gy 반화소 단위: dq4/2 -> 제곱 분모 4), st0 격자 -> gauss10 -> st 격자, x1023, log1p 부호
  const int eSt0 = (int) S("energy_st0_exp"), eSt = (int) S("energy_st_exp");
  V gx(WS * WS), gy(WS * WS);
  for (int yy = 0; yy < WS; yy++) for (int xx = 0; xx < WS; xx++) { gx[yy * WS + xx] = xd[yy * WS + clampi(xx + 1, 0, WS - 1)] - xd[yy * WS + clampi(xx - 1, 0, WS - 1)]; gy[yy * WS + xx] = xd[clampi(yy + 1, 0, WS - 1) * WS + xx] - xd[clampi(yy - 1, 0, WS - 1) * WS + xx]; }
  std::vector<V> lst(3, V(WS * WS));
  for (int c = 0; c < 3; c++)
  {
    V s0(WS * WS), s1v; for (int p = 0; p < WS * WS; p++) s0[p] = energyQ(c == 0 ? gx[p] : (c == 1 ? gx[p] : gy[p]), c == 0 ? gx[p] : gy[p], eSt0, 4);
    sepFirRep(s0, WS, WS, A("fir_gauss10"), s1v, eSt0, eSt, eSt);
    for (int p = 0; p < WS * WS; p++)
    {
      const int64_t v = s1v[p] * 1023; const int sg = v < 0 ? -1 : (v > 0 ? 1 : 0);        // |st| + 1 : 값 = |v|/2^eSt + 1
      i128 num = (i128) (v < 0 ? -v : v) + ((i128) 1 << eSt); const int64_t l10 = lnQ10(num, eSt);
      lst[c][p] = clamp16(rshr((i128) sg * l10, 2));                                           // Q8.8
    }
  }
  std::vector<V> qin; std::vector<int> qs;
  for (int c = 0; c < 16; c++) { qin.push_back(V(WS * WS, sd[c])); qs.push_back(14); }
  for (int c = 0; c < 3; c++) { qin.push_back(lst[c]); qs.push_back(8); }
  qin.push_back(lpn[3]); qs.push_back(14); for (int c = 0; c < 4; c++) { qin.push_back(cf[c]); qs.push_back(14); }
  std::vector<V> q1, q; int sq1, sq; conv1x1(qin, qs, WS * WS, "conv_dq_0", q1, sq1, true); conv1x1(q1, std::vector<int>(32, sq1), WS * WS, "conv_dq_2", q, sq, false);
  // ---- L5 후보 이동 + L6 주의 (8x8, 창 6..13)
  const Arr &dx = A("cand_dx_q4"), &dy = A("cand_dy_q4"), &phiM = A("mat_phi_i16"); const int ephi = (int) S("mat_phi_exp");
  std::vector<V> Vc(K); for (int c = 0; c < K; c++) shiftCrop(xd, WS, WS, fx + (int) dx.v[c], fy + (int) dy.v[c], 6, 8, Vc[c]);
  V ylin(64);
  for (int t = 0; t < 64; t++)
  {
    const int py = 6 + t / 8, px = 6 + t % 8, p = py * WS + px; int64_t lg[64]; int64_t mx = INT64_MIN;
    for (int c = 0; c < K; c++) { i128 acc = 0; for (int j = 0; j < 5; j++) acc += (i128) q[j][p] * phiM.v[c * 5 + j]; lg[c] = rshr(acc, sq + ephi - 8); mx = std::max(mx, lg[c]); }
    int64_t e45[64]; i128 sum = 0;
    for (int c = 0; c < K; c++) { int64_t tt; int n; expQ(lg[c] - mx, tt, n); e45[c] = rshr(tt, 15 - n - 45); sum += e45[c]; }
    int64_t r15; int e; recipQ(sum, 45, r15, e); i128 acc = 0;
    for (int c = 0; c < K; c++) { const int64_t a15 = clamp32(rshr((i128) e45[c] * r15, 45 + e)); acc += (i128) a15 * Vc[c][t]; }
    ylin[t] = rshr(acc, 15);
  }
  o.ylin = ylin;
  // ---- L7 보정: 정렬 창 16x16 (창 2..17) -> corr_in 격자, lpn 4채널 [2:18], cf[0,1,2,3,8,9] [2:18] -> conv/FiltAct x4 -> conv -> [6:10]
  V Aw; shiftCrop(xd, WS, WS, fx, fy, 2, 16, Aw); const int eIn = (int) S("corr_in_exp");
  std::vector<V> cin(11, V(256)); std::vector<int> cs(11, 14); cs[0] = eIn;
  for (int i = 0; i < 256; i++) cin[0][i] = clamp16(fdiv_round(((i128) Aw[i]) << eIn, PIX_DEN));
  auto crop = [&](const V &src, V &dst) { dst.resize(256); for (int yy = 0; yy < 16; yy++) for (int xx = 0; xx < 16; xx++) dst[yy * 16 + xx] = src[(yy + 2) * WS + xx + 2]; };
  for (int b = 0; b < 4; b++) crop(lpn[b], cin[1 + b]); const int cfi[6] = { 0, 1, 2, 3, 8, 9 }; for (int k = 0; k < 6; k++) crop(cf[cfi[k]], cin[5 + k]);
  // 분자 형식 (분모 256): 첫 층 입력은 선형이므로 x*256
  std::vector<V> num(11, V(256)); for (int c = 0; c < 11; c++) for (int i = 0; i < 256; i++) num[c][i] = cin[c][i] * 256;
  std::vector<V> c0, f0, c1, f1, c2, f2, c3, f3, c4; int sc0, sf0, sc1, sf1, sc2, sf2, sc3, sf3, sc4;
  conv3x3(num, cs, 16, 16, "conv_corr_0", c0, sc0);
  auto fa = [&](const std::vector<V> &cin_, int s_, const std::string &nm, std::vector<V> &out_, int &so_) { out_.resize(cin_.size()); for (size_t ch = 0; ch < cin_.size(); ch++) filtAct(cin_[ch], 16, 16, s_, nm, out_[ch], so_); };
  fa(c0, sc0, "filtact_corr_1", f0, sf0);
  auto tonum = [&](std::vector<V> &a, std::vector<V> &n) { n.assign(a.size(), V(256)); for (size_t ch = 0; ch < a.size(); ch++) for (int i = 0; i < 256; i++) n[ch][i] = a[ch][i] * 256; };
  std::vector<V> n0; tonum(f0, n0); conv3x3(n0, std::vector<int>(32, sf0), 16, 16, "conv_corr_2", c1, sc1); fa(c1, sc1, "filtact_corr_3", f1, sf1);
  std::vector<V> n1; tonum(f1, n1); conv3x3(n1, std::vector<int>(32, sf1), 16, 16, "conv_corr_4", c2, sc2); fa(c2, sc2, "filtact_corr_5", f2, sf2);
  std::vector<V> n2; tonum(f2, n2); conv3x3(n2, std::vector<int>(32, sf2), 16, 16, "conv_corr_6", c3, sc3); fa(c3, sc3, "filtact_corr_7", f3, sf3);
  std::vector<V> n3; tonum(f3, n3); conv3x3(n3, std::vector<int>(32, sf3), 16, 16, "conv_corr_8", c4, sc4);
  V r(16); for (int yy = 0; yy < 4; yy++) for (int xx = 0; xx < 4; xx++) r[yy * 4 + xx] = c4[0][(yy + 6) * 16 + xx + 6];   // 정규화 값 r/2^sc4
  // ---- L8 율 문턱: c = pix(D4 r D4^T), th(qp), arg = Q8.8(4(|c|-th) * recip(th)), c' = pix(c*sig), r' = pix(D4^T c' D4)
  V rq(16);
  { const Arr &D = A("mat_d4_i16"); const int eD = (int) S("mat_d4_exp"); const int64_t th = S("rate_th_q4_by_qp", clampi(qp, 0, 63));
    V c(16), c2(16);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { i128 acc = 0; for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) acc += (i128) D.v[i * 4 + k] * r[k * 4 + l] * D.v[j * 4 + l]; c[i * 4 + j] = rshr(acc * PIX_DEN, 2 * eD + sc4); }
    // recip(th): th 정규화 값 = th/PIX_DEN ; e: PIX_DEN*2^e <= th < PIX_DEN*2^(e+1)
    int e = -60; while (((i128) PIX_DEN << (e + 1 + 60)) <= ((i128) th << 60)) e++;   // e 는 음수일 수 있음
    // m10 = floor(th/(PIX_DEN*2^e) * 1024 + 0.5)
    i128 numm = (i128) th * 1024, denm = (i128) PIX_DEN; if (e >= 0) denm <<= e; else numm <<= (-e);
    int m10 = (int) fdiv_round(numm, denm); const int64_t r15 = LUT_RECIP->v[m10 - 1024];
    for (int i = 0; i < 16; i++)
    {
      const int64_t ac = c[i] < 0 ? -c[i] : c[i];
      // arg = floor( 4*(|c|-th)/PIX_DEN * r15/2^15/2^e * 256 + 0.5 )
      i128 nn = (i128) 4 * (ac - th) * r15 * 256; i128 dd = (i128) PIX_DEN << 15; if (e >= 0) dd <<= e; else nn <<= (-e);
      const int64_t arg = clamp32(fdiv_round(nn, dd)); const int64_t s15 = sigQ15(arg);
      c2[i] = rshr((i128) c[i] * s15, 15);
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { i128 acc = 0; for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++) acc += (i128) D.v[k * 4 + i] * c2[k * 4 + l] * D.v[l * 4 + j]; rq[i * 4 + j] = rshr(acc, 2 * eD); }
  }
  o.y.resize(16); for (int yy = 0; yy < 4; yy++) for (int xx = 0; xx < 4; xx++) o.y[yy * 4 + xx] = ylin[(yy + 2) * 8 + xx + 2] + rq[yy * 4 + xx];
}

int main(int argc, char **argv)
{
  if (argc < 3) { fprintf(stderr, "usage: %s export_dir dump.dmp\n", argv[0]); return 1; }
  loadExport(argv[1]);
  LUT_EXP2 = &A("lut_exp2_frac_q15"); LUT_LOG2 = &A("lut_log2_mant_q10"); LUT_RECIP = &A("lut_recip_mant_q15"); LUT_SIG = &A("lut_sigmoid_q15");
  LOG2E_Q = S("const_log2e_q16"); LN2_Q = S("const_ln2_q16"); INV30_Q = S("const_inv30_q16");
  FILE *df = fopen(argv[2], "rb"); const int REC = 1936; std::vector<int16_t> rec(REC);
  const Arr &ri = A("ref_idx"), &rxd = A("ref_xd_q4"), &ryl = A("ref_ylin_q4"), &ry = A("ref_y_q4"); const int nref = (int) ri.shape[0];
  long okx = 0, okl = 0, oky = 0; int64_t mdx = 0, mdl = 0, mdy = 0; int firstBad = -1;
  for (int k = 0; k < nref; k++)
  {
    fseek(df, (long) ri.v[k] * REC * 2, SEEK_SET); if (fread(rec.data(), 2, REC, df) != (size_t) REC) break;
    Out o; hypothesis(rec, o);
    int64_t dx = 0, dl = 0, dy = 0;
    for (int i = 0; i < 400; i++) dx = std::max(dx, std::abs(o.xd[i] - rxd.v[(int64_t) k * 400 + i]));
    for (int i = 0; i < 64; i++) dl = std::max(dl, std::abs(o.ylin[i] - ryl.v[(int64_t) k * 64 + i]));
    for (int i = 0; i < 16; i++) dy = std::max(dy, std::abs(o.y[i] - ry.v[(int64_t) k * 16 + i]));
    okx += (dx == 0); okl += (dl == 0); oky += (dy == 0); mdx = std::max(mdx, dx); mdl = std::max(mdl, dl); mdy = std::max(mdy, dy);
    if (dy != 0 && firstBad < 0) firstBad = k;
  }
  printf("ref %d | xd exact %ld (max|d| %lld) | y_lin exact %ld (max|d| %lld) | y exact %ld (max|d| %lld) | first y mismatch %d\n", nref, okx, (long long) mdx, okl, (long long) mdl, oky, (long long) mdy, firstBad);
  fclose(df); return 0;
}
