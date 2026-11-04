"""yvvyee: VTM-23.8 휘도 보간(14비트 중간 정밀도)·양방향 평균·BDOF 의 덤프 위 재현 (측정 전용).

검증:
  층 1 (양방향, BDOF 미적용): VTM 예측 = 두 14비트 가설의 평균 -> 재현이 비트 일치해야 한다 (필터 산술 검증)
  층 2 (BDOF 적용): 재현 BDOF 대 VTM 예측의 일치율·MSE. 레코드에 PU 안 위치가 없어 6x6 이웃 경계 처리를 두 변형으로 잰다.
    A: 이웃도 보간값 (PU 내부 가정)   B: 이웃 링을 정수 표본으로 (PU 경계 가정, VTM 의 BDOF 확장 규칙)
용도: C4 6절 선택지 2 — 임의의 가설 예측(모형 출력)에 VTM BDOF 를 적용해 비교.
사용: python bdof_replica.py dump1.dmp [dump2.dmp ...]
"""
import sys, numpy as np

LUMA = np.array([[0,0,0,64,0,0,0,0],[0,1,-3,63,4,-2,1,0],[-1,2,-5,62,8,-3,1,0],[-1,3,-8,60,13,-4,1,0],[-1,4,-10,58,17,-5,1,0],
    [-1,4,-11,52,26,-8,3,-1],[-1,3,-9,47,31,-10,4,-1],[-1,4,-11,45,34,-10,4,-1],[-1,4,-11,40,40,-11,4,-1],[-1,4,-10,34,45,-11,4,-1],
    [-1,4,-10,31,47,-9,3,-1],[-1,3,-8,26,52,-11,4,-1],[0,1,-5,17,58,-10,4,-1],[0,1,-4,13,60,-8,3,-1],[0,1,-3,8,62,-5,2,-1],[0,1,-2,4,63,-3,1,0]], np.int64)
ALT = np.array([0, 3, 9, 20, 20, 9, 3, 0], np.int64)
BD = 10; HEAD = 14 - BD                      # IF_INTERNAL_FRAC_BITS = 4
OFFS = 1 << 13                               # IF_INTERNAL_OFFS
C0IDX = 8                                    # 창에서 블록 좌상단 인덱스


def vtm_pred14(W, fx, fy, alt, r0, n):
    """정수 창 W (N,20,20) 에서 14비트 중간 정밀도 예측을 창 영역 [r0, r0+n) 에 대해 계산 (양방향: isLast=false).
    VTM filterHor/filterVer 와 같은 시프트·오프셋. 반환 int64 (N,n,n)."""
    N = W.shape[0]; out = np.zeros((N, n, n), np.int64)
    rows = np.arange(n) + r0; cols = np.arange(n) + r0
    cx = np.where((alt & (fx == 8))[:, None], ALT[None], LUMA[fx]); cy = np.where((alt & (fy == 8))[:, None], ALT[None], LUMA[fy])
    ki = np.arange(8) - 3
    # 가로 1단계 (isFirst): shift = 6-4 = 2, offset = -OFFS<<2 ; 행은 세로 필터에 필요한 -3..+4 확장
    rr = (rows[:, None] + ki[None]).reshape(-1)                                    # 세로 탭에 필요한 행
    rr_u = np.unique(rr)
    Xh = np.zeros((N, len(rr_u), n), np.int64)
    for a, r in enumerate(rr_u):
        src = W[:, r][:, None, :]                                                 # (N,1,20)
        idx = cols[:, None] + ki[None]                                            # (n,8)
        s = (src[:, 0][:, idx] * cx[:, None, :]).sum(-1)                          # (N,n)
        Xh[:, a] = (s + (-OFFS << 2)) >> 2 if True else s
    pos = {r: a for a, r in enumerate(rr_u)}
    both = (fx != 0) & (fy != 0); honly = (fx != 0) & (fy == 0); vonly = (fx == 0) & (fy != 0); none = (fx == 0) & (fy == 0)
    # 2단계 세로 (!isFirst, !isLast): shift 6, offset 0
    for i, r in enumerate(rows):
        taps = np.stack([Xh[:, pos[r + k]] for k in ki], 1)                        # (N,8,n)
        out[:, i] = ((taps * cy[:, :, None]).sum(1)) >> 6
    # xFrac==0 인 경우: 세로 1단계만 (isFirst): 정수 표본에 세로 필터, shift 2, offset -OFFS<<2
    if vonly.any():
        for i, r in enumerate(rows):
            src = np.stack([W[:, r + k][:, cols] for k in ki], 1)                 # (N,8,n)
            v = ((src * cy[:, :, None]).sum(1) + (-OFFS << 2)) >> 2
            out[vonly, i] = v[vonly]
    # yFrac==0 인 경우: 가로 1단계 결과가 곧 출력 (isLast=false)
    if honly.any():
        for i, r in enumerate(rows): out[honly, i] = Xh[honly, pos[r]]
    # 둘 다 0: 복사 (isFirst, !isLast): (pix<<4) - OFFS
    if none.any():
        for i, r in enumerate(rows): out[none, i] = (W[none, r][:, cols] << HEAD) - OFFS
    return out


def vtm_pred14_rect(W, fx, fy, alt, r0, h, w):
    """직사각 일반화: 정수 창 W (N,Hw,Ww) 에서 14비트 중간 정밀도 예측을 창 영역 행 [r0,r0+h), 열 [r0,r0+w) 에 대해 계산.
    vtm_pred14 와 같은 산술 (양방향 isLast=false 경로). 반환 int64 (N,h,w)."""
    N = W.shape[0]; W = W.astype(np.int64)
    cx = np.where((alt & (fx == 8))[:, None], ALT[None], LUMA[fx]); cy = np.where((alt & (fy == 8))[:, None], ALT[None], LUMA[fy])
    ki = np.arange(8) - 3
    rows = np.arange(h) + r0; cols = np.arange(w) + r0
    rr = np.arange(r0 - 3, r0 + h + 4)                                              # 세로 탭에 필요한 행 (h+7)
    # 가로 1단계 (isFirst, !isLast): shift 2, offset -OFFS<<2
    src = W[:, rr][:, :, cols[:, None] + ki[None]]                                  # (N,h+7,w,8)
    Xh = ((src * cx[:, None, None, :]).sum(-1) + (-OFFS << 2)) >> 2                 # (N,h+7,w)
    # 세로 2단계 (!isFirst, !isLast): shift 6
    taps = np.stack([Xh[:, 3 + k: 3 + k + h] for k in ki], 1)                       # (N,8,h,w)
    out = (taps * cy[:, :, None, None]).sum(1) >> 6
    honly = (fx != 0) & (fy == 0); vonly = (fx == 0) & (fy != 0); none = (fx == 0) & (fy == 0)
    if honly.any(): out[honly] = Xh[honly, 3:3 + h]
    if vonly.any():
        srcv = np.stack([W[:, rows + k][:, :, cols] for k in ki], 1)                 # (N,8,h,w)
        out[vonly] = (((srcv * cy[:, :, None, None]).sum(1) + (-OFFS << 2)) >> 2)[vonly]
    if none.any(): out[none] = (W[none][:, rows][:, :, cols] << HEAD) - OFFS
    return out


def uni_from14(s14):
    """단방향(isLast=true) 화소 출력. VTM 의 마지막 단계 시프트·클리핑과 비트 동일 (유도: 하위 비트가 자리올림을 만들지 못함)."""
    return np.clip((s14 + OFFS + 8) >> 4, 0, 1023)


def bi_avg(s0, s1):
    shift = HEAD + 1; off = (1 << (shift - 1)) + 2 * OFFS
    return np.clip((s0 + s1 + off) >> shift, 0, 1023)


def floor_log2(x):
    x = np.asarray(x, np.int64); return np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int64), 0)


def bdof4x4(p0, p1, g0x, g0y, g1x, g1y):
    """p0,p1: 14비트 예측 6x6 (블록 -1..4), g*: 기울기 6x6. VTM calcBIOSumsCore + xAddBIOAvg4 (4x4 출력)."""
    tGX = (g0x + g1x) >> 1; tGY = (g0y + g1y) >> 1; tDI = (p1 >> 4) - (p0 >> 4)
    sAGX = np.abs(tGX).sum((1, 2)); sAGY = np.abs(tGY).sum((1, 2))
    sDIX = np.where(tGX < 0, -tDI, np.where(tGX == 0, 0, tDI)).sum((1, 2)); sDIY = np.where(tGY < 0, -tDI, np.where(tGY == 0, 0, tDI)).sum((1, 2))
    sSGN = np.where(tGY < 0, -tGX, np.where(tGY == 0, 0, tGX)).sum((1, 2))
    tmpx = np.where(sAGX == 0, 0, (4 * sDIX) >> floor_log2(sAGX)); tmpx = np.clip(tmpx, -15, 15)
    tmpD = (sSGN * tmpx) >> 1
    tmpy = np.where(sAGY == 0, 0, (4 * sDIY - tmpD) >> floor_log2(sAGY)); tmpy = np.clip(tmpy, -15, 15)
    c = slice(1, 5)
    b = tmpx[:, None, None] * (g0x[:, c, c] - g1x[:, c, c]) + tmpy[:, None, None] * (g0y[:, c, c] - g1y[:, c, c])
    shift = HEAD + 1; off = (1 << (shift - 1)) + 2 * OFFS
    return np.clip((p0[:, c, c] + p1[:, c, c] + b + off) >> shift, 0, 1023), tmpx, tmpy


def grads(p8):
    """8x8 (블록 -2..5) 14비트 예측에서 6x6 (블록 -1..4) 기울기: (s[x+1]>>6) - (s[x-1]>>6)."""
    s = p8 >> 6
    gx = s[:, 1:7, 2:8] - s[:, 1:7, 0:6]; gy = s[:, 2:8, 1:7] - s[:, 0:6, 1:7]
    return gx, gy


def bdof_from_windows(W0, W1, fr, alt, variant='A'):
    """변형 A: 8x8 이웃까지 모두 보간값. 변형 B: 6x6 링(블록 밖)을 정수 표본(최근접, VTM 확장 규칙)으로 두고 기울기 링은 복제."""
    N = W0.shape[0]
    P = []
    for h, Wh in enumerate((W0, W1)):
        fx, fy = fr[:, 2 * h], fr[:, 2 * h + 1]
        p8 = vtm_pred14(Wh, fx, fy, alt[:, h], C0IDX - 2, 8)
        if variant == 'B':
            # 링(블록 -1..4 의 바깥 테두리)을 최근접 정수 표본으로: xOffset = (fx<8)?1:0 -> 정수 위치 -1 쪽
            xo = np.where(fx < 8, 1, 0); yo = np.where(fy < 8, 1, 0)
            p6 = p8[:, 1:7, 1:7].copy()
            for n_ in range(N):
                r0 = C0IDX - yo[n_]; c0 = C0IDX - xo[n_]                          # 링 (y,x) -> 정수 표본 (y+1-yo, x+1-xo), 링 좌상단 y=x=-1
                ring = (Wh[n_, r0:r0 + 6, c0:c0 + 6].astype(np.int64) << HEAD) - OFFS
                p6[n_, 0, :] = ring[0, :]; p6[n_, 5, :] = ring[5, :]; p6[n_, :, 0] = ring[:, 0]; p6[n_, :, 5] = ring[:, 5]
            # 기울기: 내부 4x4 는 링 포함 6x6 에서 계산, 링의 기울기는 복제 (gradFilterCore PAD)
            s = p6 >> 6
            gx = np.zeros((N, 6, 6), np.int64); gy = np.zeros((N, 6, 6), np.int64)
            gx[:, 1:5, 1:5] = s[:, 1:5, 2:6] - s[:, 1:5, 0:4]; gy[:, 1:5, 1:5] = s[:, 2:6, 1:5] - s[:, 0:4, 1:5]
            gx[:, 1:5, 0] = gx[:, 1:5, 1]; gx[:, 1:5, 5] = gx[:, 1:5, 4]; gx[:, 0] = gx[:, 1]; gx[:, 5] = gx[:, 4]
            gy[:, 1:5, 0] = gy[:, 1:5, 1]; gy[:, 1:5, 5] = gy[:, 1:5, 4]; gy[:, 0] = gy[:, 1]; gy[:, 5] = gy[:, 4]
            # 합 계산용 링은 예측값 복제 (applyBiOptFlow 의 padStr)
            q6 = p6.copy(); q6[:, 1:5, 0] = q6[:, 1:5, 1]; q6[:, 1:5, 5] = q6[:, 1:5, 4]; q6[:, 0] = q6[:, 1]; q6[:, 5] = q6[:, 4]
            P.append((q6, gx, gy))
        else:
            gx, gy = grads(p8); P.append((p8[:, 1:7, 1:7], gx, gy))
    (p0, g0x, g0y), (p1, g1x, g1y) = P
    return bdof4x4(p0, p1, g0x, g0y, g1x, g1y)


def load(files, REC=1936):
    A = np.concatenate([np.fromfile(f, dtype=np.int16).reshape(-1, REC) for f in files])
    M = A[:, :10].astype(np.int64); W = A[:, 330:1130].reshape(-1, 2, 20, 20).astype(np.int64); V = A[:, 314:330].reshape(-1, 4, 4).astype(np.int64)
    T = A[:, 298:314].reshape(-1, 4, 4).astype(np.int64); H2 = A[:, 1130:1136].astype(np.int64)
    C = np.stack([A[:, 10:154].reshape(-1, 12, 12)[:, 4:8, 4:8], A[:, 154:298].reshape(-1, 12, 12)[:, 4:8, 4:8]], 1).astype(np.int64)
    return M, W, V, T, H2, C


if __name__ == '__main__':
    M, W, V, T, H2, C = load(sys.argv[1:])
    st = M[:, 0]; fr = np.maximum(M[:, 5:9], 0); imv = H2[:, 0]; alt = np.stack([(imv == 3) & (fr[:, 0] == 8) | (imv == 3) & (fr[:, 1] == 8)] * 2, 1)
    alt = np.stack([imv == 3, imv == 3], 1)          # 대체 필터는 imv=3 이고 해당 축 분수 8 일 때 (vtm_pred14 안에서 축별 판정)
    for s_, nm in ((1, '층 1 (평균)'), (2, '층 2 (BDOF)')):
        m = st == s_
        if not m.any(): continue
        W0, W1, f, a, v, c = W[m, 0], W[m, 1], fr[m], alt[m], V[m], C[m]
        p0 = vtm_pred14(W0, f[:, 0], f[:, 1], a[:, 0], C0IDX, 4); p1 = vtm_pred14(W1, f[:, 2], f[:, 3], a[:, 1], C0IDX, 4)
        avg = bi_avg(p0, p1)
        # 덤프의 DCTIF 저장값(x4)과 14비트 재현의 일치 확인 (가설 0): 저장값 = round(실수값*4), 14비트 = 실수값*16 - 8192 (반올림 차이 허용)
        d_c = np.abs((p0 + OFFS) / 16.0 - c[:, 0] / 4.0)
        print(f'{nm}: n={m.sum()} | 가설0 14비트 재현 대 덤프 DCTIF(x4) 최대 차 {d_c.max():.3f} 화소, 평균 {d_c.mean():.4f}')
        print(f'   단순 평균 재현 대 VTM 예측: 완전 일치 블록 {np.all(avg == v, axis=(1, 2)).mean():.4f}, 화소 일치 {(avg == v).mean():.4f}, 최대 차 {np.abs(avg - v).max()}')
        if s_ == 2:
            for var in ('A', 'B'):
                out, tx, ty = bdof_from_windows(W0, W1, f, a, var)
                print(f'   BDOF 재현 {var} 대 VTM 예측: 완전 일치 블록 {np.all(out == v, axis=(1, 2)).mean():.4f}, 화소 일치 {(out == v).mean():.4f}, '
                      f'MSE 재현-VTM {((out - v) ** 2).mean():.3f} | SSE 대 원본: VTM {((v - T[m]) ** 2).sum() / ((avg - T[m]) ** 2).sum() * 100 - 100:+.3f} %, 재현 {((out - T[m]) ** 2).sum() / ((avg - T[m]) ** 2).sum() * 100 - 100:+.3f} % (평균 대비) | v≠0 비율 {((tx != 0) | (ty != 0)).mean():.3f}')
