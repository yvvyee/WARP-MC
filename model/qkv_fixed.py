"""yvvyee: QKV-MC 고정소수 모사(fake quantization) — stage4_integration.md 4.3절의 형식을 부동소수 그래프 위에 양자화 지점으로 올려
정수화 손실(출력 차, 덤프 SSE 차)을 잰다. C++ 비트 일치 구현의 전 단계다.

형식 (Q 객체로 켜고 끔):
  w16   : 가중치·FIR 계수 int16 (텐서별 2의 거듭제곱 스케일, FIR 은 Q1.14)
  a_q88 : MLP·합성곱 활성 Q8.8 (1/256 격자, ±128 포화)
  pix   : 화소 영역 값 Q10.4 (1/16 격자, 입력 창·대역·이동·예측)
  lut   : exp / sigmoid / log2 / 역수를 조회표(입력 격자 양자화 + 출력 Q0.15 / Q1.15)로 모사
  lrelu : LeakyReLU 음수 기울기 51/256
사용: python qkv_fixed.py ckpt.pt [dump.dmp ...]
     python qkv_fixed.py --tile ckpt.pt tile.dmp export_dir [ref_out.bin] --f64   (yvvyee: 타일 레코드, 내보내기 스케일 사용)
"""
import math, sys, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import qkv_mc as Mfl
from qkv_mc import WS, QKVMC, gauss1d, precision_onehot, codemap_feats


class Quant:
    GROUPS = ('fir', 'weight', 'act', 'pix', 'lut', 'lrelu', 'energy')

    def __init__(self, on=True):
        self.on = on; self.groups = set(self.GROUPS)
        self.w_bits = 16; self.fir_frac = 14; self.act_frac = 8; self.pix_frac = 4; self.lut_in_frac = 8; self.lut_out_frac = 15
        self.overflow = {}        # 누산 범위 점검용 (|pre-activation| 최대)
        self.calib = False; self.amax = {}; self.scale = {}   # 층별 활성 스케일 보정 (1차 통과에서 최대값 수집, 2차에서 2의 거듭제곱 스케일 고정)

    def g(self, name): return self.on and name in self.groups

    def rnd(self, x, frac, bits=16):
        if not self.on: return x
        s = 2.0 ** frac; lim = 2.0 ** (bits - 1) - 1
        return torch.clamp(torch.floor(x * s + 0.5), -lim, lim) / s     # floor(x+0.5): C++ 의 (acc+half)>>shift 와 같음

    def weight(self, w):
        """텐서별 2의 거듭제곱 스케일 int16 대칭 양자화."""
        if not self.g('weight'): return w
        m = w.abs().max().clamp(min=1e-12); e = torch.floor(torch.log2(32767.0 / m)); s = 2.0 ** e
        return torch.clamp(torch.floor(w * s + 0.5), -32767, 32767) / s

    def fir(self, h):
        if not self.g('fir'): return h
        s = 2.0 ** self.fir_frac; q = torch.floor(h * s + 0.5)
        q[len(q) // 2] += s - q.sum()                      # 합을 정확히 1 (Q1.14) 로 보정
        return q / s

    def act(self, x, key=None, frac=14):
        if not self.g('act') and not self.calib: return x
        if key is None: return self.rnd(x, frac)                             # 유계 특징: 기본 Q1.14, 넓은 범위는 frac 지정
        if self.calib:
            self.amax[key] = max(self.amax.get(key, 0.0), float(x.abs().max())); return x
        sc = self.scale.get(key)
        if sc is None: sc = 2.0 ** torch.floor(torch.log2(torch.tensor(32767.0 / max(self.amax.get(key, 1.0), 1e-9)))).item(); self.scale[key] = sc
        return torch.clamp(torch.floor(x * sc + 0.5), -32767, 32767) / sc    # 층별 int16 (보정된 2의 거듭제곱 스케일)
    def pix(self, x):
        """화소 영역 Q10.4: 정규화 값(1/1023 단위)에 10비트 눈금의 1/16 격자, int16 범위(±2048 화소) 포화."""
        if not self.g('pix'): return x
        sc = 1024.0 * 2.0 ** self.pix_frac                                  # 정규화 1.0 = 1024 화소 (2진 격자, C++ 비트 일치)
        return torch.clamp(torch.floor(x * sc + 0.5), -32767, 32767) / sc

    def energy(self, x, key):
        """에너지(제곱합) 텐서: 보정된 2의 거듭제곱 스케일의 int32."""
        if self.calib:
            self.amax[('e', key)] = max(self.amax.get(('e', key), 0.0), float(x.abs().max())); return x
        if not self.g('energy'): return x
        sc = self.scale.get(('e', key))
        if sc is None: sc = min(2.0 ** torch.floor(torch.log2(torch.tensor(2147483647.0 / max(self.amax.get(('e', key), 1.0), 1e-30)))).item(), 2.0 ** 40); self.scale[('e', key)] = sc
        return torch.clamp(torch.floor(x * sc + 0.5), -2147483647, 2147483647) / sc

    # 조회표 모사: 입력을 조회표 격자로 양자화한 뒤 함수값을 출력 격자로 양자화
    def lut(self, fn, x, in_frac=None, out_frac=None, in_lim=None):
        if not self.g('lut'): return fn(x)
        xi = self.rnd(x, self.lut_in_frac if in_frac is None else in_frac, bits=32)
        if in_lim is not None: xi = xi.clamp(-in_lim, in_lim)
        return self.rnd(fn(xi), self.lut_out_frac if out_frac is None else out_frac, bits=32)

    LOG2E_Q = 2.0 ** -16 * math.floor(math.log2(math.e) * 2 ** 16 + 0.5)      # log2(e) 를 Q1.16 상수로
    LN2_Q = 2.0 ** -16 * math.floor(math.log(2.0) * 2 ** 16 + 0.5)            # ln2 를 Q0.16 상수로

    def exp(self, x):      # e^x = 2^(x*log2e): 정수부는 시프트, 소수부 256항목 조회표 (C++ 재현 가능)
        if not self.g('lut'): return torch.exp(x)
        t = self.rnd(self.rnd(x, 8, bits=32) * self.LOG2E_Q, 8, bits=32).clamp(-86, 86)
        n = torch.floor(t); f = t - n
        return self.rnd(torch.exp2(f), 15, bits=32) * torch.exp2(n)
    def sigmoid(self, x): return self.lut(torch.sigmoid, x, in_frac=8, out_frac=15, in_lim=16)
    def log(self, x):      # 가수·지수 정규화 후 log2 조회표(1024항목), ln2 상수(Q0.16) 곱, 결과 Q?.10
        if not self.g('lut'): return torch.log(x)
        x = x.clamp(min=1e-30); e = torch.floor(torch.log2(x)); m = x / 2.0 ** e            # m in [1,2)
        l2 = e + self.rnd(torch.log2(self.rnd(m, 10, bits=32)), 10, bits=32)
        return self.rnd(l2 * self.LN2_Q, 10, bits=32)
    def recip(self, x):    # 정규화 가수의 역수 조회표 (Q1.15) 와 지수 시프트
        if not self.g('lut'): return 1.0 / x
        x = x.clamp(min=1e-30); e = torch.floor(torch.log2(x)); m = x / 2.0 ** e
        return self.rnd(1.0 / self.rnd(m, 10, bits=32), 15, bits=32) / 2.0 ** e
    def lrelu(self, x):
        return torch.where(x > 0, x, x * (51.0 / 256 if self.g('lrelu') else 0.2))


Q = Quant(True)


def sep_blur_q(x, h, ekey=None):
    """ekey 가 없으면 화소 텐서(Q10.4), 있으면 에너지 텐서(보정 int32)로 양자화."""
    qz = (lambda t: Q.pix(t)) if ekey is None else (lambda t: Q.energy(t, ekey))
    h = Q.fir(h); c = x.shape[1]; r = (len(h) - 1) // 2
    kx = h.view(1, 1, 1, -1).repeat(c, 1, 1, 1).to(x); ky = h.view(1, 1, -1, 1).repeat(c, 1, 1, 1).to(x)
    x = qz(F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), kx, groups=c))
    return qz(F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), ky, groups=c))


def kaiser_frac_q(f, taps=12, beta=6.0):
    h = Mfl.kaiser_frac(f, taps, beta)
    if Q.on:
        s = 2.0 ** Q.fir_frac; q = torch.floor(h * s + 0.5); q[..., taps // 2 - 1] += s - q.sum(-1)    # 위상별 표, 합 1 보정
        h = q / s
    return h


def shift_crop2_q(X, sx, sy, r0, ny, nx, taps=12):
    """yvvyee: 직사각형 영역 [r0, r0+ny) x [r0, r0+nx) 의 창 sinc 이동 (qkv_mc.shift_crop2 의 양자화판). nx=ny 이면 이전 정사각형판과 같은 연산."""
    N, H, W = X.shape; dev = X.device; kk = torch.arange(taps, device=dev) - (taps // 2 - 1)
    def axis(s, L, n):
        pos = (torch.arange(n, device=dev, dtype=X.dtype) + r0)[None] + s[:, None]; i0 = torch.floor(pos); f = pos - i0
        return (i0.long()[..., None] + kk).clamp(0, L - 1), kaiser_frac_q(f, taps)
    ix, wx = axis(sx, W, nx); iy, wy = axis(sy, H, ny)
    g = torch.gather(X[:, :, None, :].expand(N, H, nx, W), 3, ix[:, None].expand(N, H, nx, taps)); Xh = Q.pix((g * wx[:, None]).sum(-1))
    g = torch.gather(Xh[:, None].expand(N, ny, H, nx), 2, iy[..., None].expand(N, ny, taps, nx)); return Q.pix((g * wy[..., None]).sum(2))


def shift_crop_q(X, sx, sy, r0, n, taps=12): return shift_crop2_q(X, sx, sy, r0, n, n, taps)


def conv_q(m, x, pad_mode=None):
    """nn.Conv2d 를 양자화 가중치·Q8.8 활성으로 실행 (반사 패딩 유지)."""
    w = Q.weight(m.weight); b = Q.act(m.bias) if m.bias is not None else None
    if m.padding_mode == 'reflect' and m.padding[0] > 0:
        x = F.pad(x, (m.padding[0],) * 4, mode='reflect'); y = F.conv2d(x, w, b)
    else:
        y = F.conv2d(x, w, b, padding=m.padding)
    Q.overflow[id(m)] = max(Q.overflow.get(id(m), 0.0), float(y.abs().max()))
    return Q.act(y, key=id(m))


def filtact_q(m, x):
    b, c, h, w = x.shape
    u = x.new_zeros(b, c, 2 * h, 2 * w); u[:, :, ::2, ::2] = x
    hu = Q.fir(m.hu / 2) * 2; hd = Q.fir(m.hd)
    def sep(x, hh):
        t = len(hh); kx = hh.view(1, 1, 1, t).repeat(c, 1, 1, 1).to(x); ky = hh.view(1, 1, t, 1).repeat(c, 1, 1, 1).to(x)
        x = Q.act(F.conv2d(F.pad(x, (t // 2, t // 2 - 1, 0, 0), mode='reflect'), kx, groups=c), key=(id(m), 'h'))
        return Q.act(F.conv2d(F.pad(x, (0, 0, t // 2, t // 2 - 1), mode='reflect'), ky, groups=c), key=(id(m), 'v'))
    u = Q.lrelu(sep(u, hu))
    return sep(u, hd)[:, :, ::2, ::2]


def seq_q(seq, x):
    for m in seq:
        if isinstance(m, nn.Conv2d): x = conv_q(m, x)
        elif isinstance(m, nn.ReLU): x = F.relu(x)
        elif isinstance(m, nn.LeakyReLU): x = Q.lrelu(x)
        elif m.__class__.__name__ == 'FiltAct': x = filtact_q(m, x)
        else: raise RuntimeError(m)
    return x


class HypQ:
    def __init__(self, h): self.h = h

    def bands(self, x):
        g = [sep_blur_q(x, gauss1d(s, int(math.ceil(3 * s)))) for s in self.h.SIG]
        return [g[2], Q.pix(g[1] - g[2]), Q.pix(g[0] - g[1]), Q.pix(x - g[0])]

    def forward(self, W, fr, qp, cm, imv, merge, dpoc, bsz, st, tw=4, th=4):
        """yvvyee: 타일 의미(stage4 11절) — 창 (th+16)x(tw+16), 출력 th x tw. tw=th=4 는 20x20 창의 블록 의미와 같은 연산."""
        h = self.h; N = W.shape[0]; Hh, Ww = W.shape[-2], W.shape[-1]; assert Hh == th + 16 and Ww == tw + 16 and tw % 4 == 0 and th % 4 == 0
        x = Q.pix(W[:, None])                                                     # 화소 Q10.4
        sd = Q.act(h.side(qp, fr, imv, merge, dpoc, bsz, st, max(Hh, Ww))[..., :Hh, :Ww]); cf = Q.act(codemap_feats(cm, qp))
        B = self.bands(x)
        phi_s = torch.cat([sep_blur_q(Q.energy(b * b, ('sq', i)), gauss1d(1.0, 2), ekey=('phi', i)) for i, b in enumerate(B)], 1)
        lphi = Q.act(seq_q(h.noise, torch.cat([sd, cf], 1)) - 30.0, frac=8)      # log Phi_nn, 범위 수십 -> Q8.8
        phi_n = Q.energy(Q.exp(lphi), 'phin')
        den = phi_s + phi_n + 2.0 ** -40                                             # eps 는 2의 거듭제곱 (정수 재현)
        G = Q.rnd(phi_s * Q.recip(den), 15, bits=32)                             # Wiener 비 Q0.15
        xd = Q.pix(sum(G[:, b:b + 1] * B[b] for b in range(h.nb))[:, 0])
        k = torch.cat([h.alt_k, h.alt_k[:3].flip(0)]); k = Q.fir(k / k.sum())
        g_alt = Q.rnd(torch.sigmoid(h.alt_g), 15, bits=32)
        ax = ((imv == 3) & (fr[:, 0] == 8)).to(xd.dtype).view(-1, 1, 1) * g_alt; ay = ((imv == 3) & (fr[:, 1] == 8)).to(xd.dtype).view(-1, 1, 1) * g_alt
        lx = Q.pix(F.conv2d(F.pad(xd[:, None], (3, 3, 0, 0), mode='replicate'), k.view(1, 1, 1, 7))[:, 0]); xd = Q.pix(xd + ax * (lx - xd))
        ly = Q.pix(F.conv2d(F.pad(xd[:, None], (0, 0, 3, 3), mode='replicate'), k.view(1, 1, 7, 1))[:, 0]); xd = Q.pix(xd + ay * (ly - xd))
        gx = F.pad(xd[:, :, 2:] - xd[:, :, :-2], (1, 1, 0, 0), mode='replicate') / 2; gy = F.pad(xd[:, 2:] - xd[:, :-2], (0, 0, 1, 1), mode='replicate') / 2
        stt = sep_blur_q(Q.energy(torch.stack([gx * gx, gx * gy, gy * gy], 1), 'st0'), gauss1d(1.0, 2), ekey='st') * 1023.0
        INV30 = 2.0 ** -16 * math.floor(2 ** 16 / 30.0 + 0.5)
        lpn = Q.act(Q.log(phi_n + 2.0 ** -40) * INV30)                            # 원 모형의 1e-12 포화를 2^-40 으로, 1/30 은 Q0.16 상수
        lst = Q.act(Q.log(stt.abs() + 1.0) * stt.sign(), frac=8)                # log1p(|st|) sign(st), 범위 ±16 -> Q8.8
        q = seq_q(h.dq, torch.cat([sd, lst, lpn[:, 3:4], cf[:, 0:4]], 1))
        s_x = fr[:, 0] / 16; s_y = fr[:, 1] / 16; K = len(h.dx)
        Xr = xd[:, None].expand(N, K, Hh, Ww).reshape(N * K, Hh, Ww)
        V = shift_crop2_q(Xr, (s_x[:, None] + h.dx[None]).reshape(-1), (s_y[:, None] + h.dy[None]).reshape(-1), 6, th + 4, tw + 4).view(N, K, th + 4, tw + 4)   # yvvyee: 타일+2 여유
        logit = Q.rnd(torch.einsum('nchw,kc->nkhw', q[:, :, 6:6 + th + 4, 6:6 + tw + 4], Q.weight(h.phi)), 8, bits=32)   # Q8.8, 포화 없음(int32)
        e = Q.rnd(Q.exp(logit - logit.max(1, keepdim=True).values), 45, bits=64); alpha = Q.rnd(e * Q.recip(e.sum(1, keepdim=True)), 15, bits=32)
        y_lin = Q.pix((alpha * V).sum(1))
        A = shift_crop2_q(xd, s_x, s_y, 2, th + 12, tw + 12)                       # yvvyee: 정렬 창 = 타일+12
        Aq = Q.act(A, key='corr_in')                                               # 화소 창을 2진 격자(층별 int16)로
        ch, cw = slice(2, 2 + th + 12), slice(2, 2 + tw + 12)
        r = seq_q(h.corr, torch.cat([Aq[:, None], lpn[:, :, ch, cw], cf[:, [0, 1, 2, 3, 8, 9], ch, cw]], 1))[:, 0, 6:6 + th, 6:6 + tw]
        if h.rate_gate:   # yvvyee: 타일 안 4x4 블록별 DCT 문턱 (블록 차원 (bh, bw) 를 배치처럼 둔다)
            step = (4 * 2 ** ((qp - 4) / 6) / PIXN).view(-1, 1, 1, 1, 1); D4 = Q.weight(h.D4)
            rb = r.reshape(N, th // 4, 4, tw // 4, 4).permute(0, 1, 3, 2, 4)
            c = Q.pix(D4 @ rb @ D4.T); thr = Q.pix(torch.exp(h.tau) * step)
            arg = Q.rnd(4.0 * (c.abs() - thr) * Q.recip(thr), 8, bits=32)
            c = Q.pix(c * Q.sigmoid(arg)); r = Q.pix(D4.T @ c @ D4).permute(0, 1, 3, 2, 4).reshape(N, th, tw)
        return dict(y=Q.pix(y_lin[:, 2:2 + th, 2:2 + tw] + r), y_lin=y_lin, phi_s=phi_s, phi_n=phi_n, lphi=lphi, xd=xd, r=r)


class NetQ:
    def __init__(self, net): self.net = net; self.hq = HypQ(net.h)

    def forward(self, b):
        net = self.net; W, fr, qp, bi, cm = b['W'], b['fr'].clamp(min=0), b['qp'], b['bi'], b['cm']
        imv, merge, dpoc, bsz, st = b['imv'], b['merge'], b['dpoc'], b['bsz'], b['st']
        tw, th = b.get('tw', 4), b.get('th', 4)
        if (tw, th) != (4, 4): assert net.no_l9, 'yvvyee: L9 경로는 4x4 블록 의미에서만 구현됨'
        p0 = self.hq.forward(W[:, 0], fr[:, 0:2], qp, cm[:, 0], imv, merge, dpoc[:, 0], bsz, st, tw=tw, th=th)
        if not bool(bi.any()): return p0['y']
        ib = bi.nonzero()[:, 0]
        p1 = self.hq.forward(W[ib, 1], fr[ib, 2:4], qp[ib], cm[ib, 1], imv[ib], merge[ib], dpoc[ib, 1], bsz[ib], st[ib], tw=tw, th=th)
        if net.no_l9:
            out = p0['y'].clone(); out[ib] = Q.pix(0.5 * (p0['y'][ib] + p1['y'])); return out
        y0 = p0['y_lin'][ib]; y1 = p1['y_lin']; nb = net.h.nb
        B0 = self.hq.bands(y0[:, None]); B1 = self.hq.bands(y1[:, None])
        dphi = (fr[ib, 0:2] - fr[ib, 2:4]) * (2 * math.pi / 16)
        d0, d1 = dpoc[ib, 0].long(), dpoc[ib, 1].long(); w4, h4 = bsz[ib, 0].long(), bsz[ib, 1].long()
        elig = ((d0 * d1 < 0) & (d0.abs() == d1.abs()) & (w4 * h4 * 16 >= 128) & (torch.minimum(w4, h4) * 4 >= 8)).float()
        smvd = (elig * (st[ib] == 1).float())[:, None]
        cond = Q.act(torch.cat([Q.lut(torch.sin, dphi, 8, 15), Q.lut(torch.cos, dphi, 8, 15), qp[ib, None] / 63.0, dpoc[ib].float() / 16.0, F.one_hot(st[ib].long().clamp(0, 2), 3).float()[:, 1:], smvd], 1))[:, :, None, None].expand(-1, -1, 8, 8)
        stats = Q.act(torch.cat([Q.log(p0['phi_s'][ib][:, :, 6:14, 6:14] + 1e-12), Q.log(p1['phi_s'][:, :, 6:14, 6:14] + 1e-12), torch.log(p0['phi_n'][ib][:, :, 6:14, 6:14]), torch.log(p1['phi_n'][:, :, 6:14, 6:14])], 1) / 30.0)
        xin = torch.cat([cond, stats], 1); sel = (st[ib] == 2).float().view(-1, 1, 1, 1)
        if net.pd_gate:
            sx = Q.lut(torch.sin, dphi[:, 0] / 2, 8, 15).abs(); sy = Q.lut(torch.sin, dphi[:, 1] / 2, 8, 15).abs(); gpd = (1 - (1 - sx) * (1 - sy)).view(-1, 1, 1, 1)
        else: gpd = torch.ones(len(ib), 1, 1, 1, device=W.device, dtype=W.dtype)
        logit = sel * seq_q(net.comb[1], xin) + (1 - sel) * seq_q(net.comb[0], xin)
        beta = Q.rnd(0.5 + (Q.sigmoid(logit) - 0.5) * gpd, 15, bits=32)
        yb = Q.pix(sum(beta[:, k:k + 1] * B0[k] + (1 - beta[:, k:k + 1]) * B1[k] for k in range(nb))[:, 0])
        din = torch.stack([y0, y1, Q.pix(y0 - y1)], 1)
        rb = Q.pix((sel * seq_q(net.bcorr[1], din) + (1 - sel) * seq_q(net.bcorr[0], din))[:, 0, 2:6, 2:6] * gpd[:, 0])
        r0 = p0['y'][ib] - p0['y_lin'][ib][:, 2:6, 2:6]; r1 = p1['y'] - y1[:, 2:6, 2:6]
        out = p0['y'].clone(); out[ib] = Q.pix(yb[:, 2:6, 2:6] + 0.5 * (r0 + r1) + rb)
        return out


# ---------------------------------------------------------------- 실행: 부동소수 대 고정소수 모사 비교
def load_dump(files, REC=1936):
    M, W, T, C, H2, CM = [], [], [], [], [], []
    for f in files:
        a = np.fromfile(f, dtype=np.int16).reshape(-1, REC)
        M.append(a[:, :10]); W.append(a[:, 330:1130].reshape(-1, 2, 20, 20)); T.append(a[:, 298:314].reshape(-1, 4, 4))
        C.append(np.stack([a[:, 10:154].reshape(-1, 12, 12)[:, 4:8, 4:8], a[:, 154:298].reshape(-1, 12, 12)[:, 4:8, 4:8]], 1)); H2.append(a[:, 1130:1136]); CM.append(a[:, 1136:1936].reshape(-1, 2, 20, 20))
    M = np.concatenate(M); W = np.concatenate(W); T = np.concatenate(T).astype(np.float32); C = np.concatenate(C).astype(np.float32) / 4; H2 = np.concatenate(H2); CM = np.concatenate(CM)
    st = M[:, 0].astype(np.int64); bi = st > 0; C0 = np.where(bi[:, None, None], C.mean(1), C[:, 0])
    return M, W, T, C0, H2, CM, st, bi


DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64 if '--f64' in sys.argv else torch.float32      # 비트 일치 기준 출력은 float64 (int16 곱의 합이 정확)
def ckpt_ch(sd): return int(sd['h.corr.0.weight'].shape[0])   # yvvyee: L7 채널 수 (CH=16·32 학습 체크포인트 모두 적재)
PIXN = 1024.0                                                     # 모사기의 화소 정규화 상수 (부동소수 모형은 1023)


def batches(M, W, H2, CM, st, bi, bs=512):
    for i in range(0, len(M), bs):
        j = slice(i, min(len(M), i + bs)); h = H2[j].astype(np.int64); g = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
        yield j, dict(W=g(W[j].astype(np.float64)).to(DT) / PIXN, fr=g(M[j, 5:9].astype(np.float64)).to(DT), qp=g(M[j, 1].astype(np.float64)).to(DT), bi=g(bi[j]),
                      imv=g(h[:, 0]), merge=g(h[:, 1]), dpoc=g(h[:, 3:5]), bsz=g(np.stack([h[:, 5] >> 8, h[:, 5] & 255], 1)), cm=g(CM[j].astype(np.int32)), st=g(st[j]))


def stream_eval(ck, files, calib_files):
    """파일 단위 스트리밍: 보정은 calib_files 의 앞 4096 레코드, 평가는 files 전체를 층별 SSE 합으로 누적 (메모리 최소)."""
    import os
    sd = torch.load(ck, map_location='cpu'); net = QKVMC(no_l9=('--no-l9' in sys.argv), ch=ckpt_ch(sd)); net.load_state_dict(sd); net.eval(); net.to(DEV)
    if DT == torch.float64: net.double()
    nq = NetQ(net)
    with torch.no_grad():
        Q.calib = True; n = 0
        for f in calib_files:
            M, W, T, C0, H2, CM, st, bi = load_dump([f])
            for j, b in batches(M, W, H2, CM, st, bi):
                nq.forward(b); n += j.stop - j.start
                if n >= 4096: break
            if n >= 4096: break
        Q.calib = False
        acc = {}
        for f in files:
            M, W, T, C0, H2, CM, st, bi = load_dump([f]); cls = os.path.basename(f)[0]
            Pf = np.zeros((len(M), 4, 4), np.float32); Pq = np.zeros_like(Pf)
            for j, b in batches(M, W, H2, CM, st, bi):
                bf = dict(b); bf['W'] = b['W'] * (PIXN / 1023.0); Pf[j] = (net(bf) * 1023).cpu().numpy(); Pq[j] = (nq.forward(b) * PIXN).cpu().numpy()
            for s_ in (0, 1, 2):
                m = st == s_
                if not m.any(): continue
                for key in ((s_, 'all'), (s_, cls)):
                    a = acc.setdefault(key, [0.0, 0.0, 0.0, 0, 0.0, 0.0])
                    a[0] += ((T[m] - C0[m]) ** 2).sum(); a[1] += ((T[m] - Pf[m]) ** 2).sum(); a[2] += ((T[m] - Pq[m]) ** 2).sum(); a[3] += int(m.sum())
                    a[4] += np.abs(Pq[m] - Pf[m]).sum(); a[5] = max(a[5], float(np.abs(Pq[m] - Pf[m]).max()))
            print(f'{os.path.basename(f)} done', flush=True)
    print(f'{"subset":12s} {"n":>8s} {"float":>8s} {"fixed":>8s} {"mean|d|":>8s} {"max|d|":>7s}')
    for key in sorted(acc, key=lambda k: (k[0], k[1] != 'all', k[1])):
        a = acc[key]; print(f'{str(key):12s} {a[3]:8d} {100 * (a[1] / a[0] - 1):+8.3f} {100 * (a[2] / a[0] - 1):+8.3f} {a[4] / a[3] / 16:8.4f} {a[5]:7.2f}')


# ---------------------------------------------------------------- yvvyee: 타일 레코드(4640 int16, stage4 11.10절) 모사와 C++ 비트 일치 기준 출력
TREC, TWIN, TOUT = 4640, 32, 16


def load_tile(files):
    """yvvyee: 머리부 32, 가설별 32x32 창 2, 가설별 32x32 지도 2, 16x16 원본, 16x16 VTM 예측."""
    a = np.concatenate([np.fromfile(f, dtype=np.int16).reshape(-1, TREC) for f in files])
    Hd = a[:, :32].astype(np.int64); Wn = a[:, 32:2080].reshape(-1, 2, TWIN, TWIN); CMt = a[:, 2080:4128].reshape(-1, 2, TWIN, TWIN)
    return Hd, Wn, CMt


def tile_groups(Hd):
    g = {}
    for i, (w, h) in enumerate(zip(Hd[:, 9], Hd[:, 10])): g.setdefault((int(w), int(h)), []).append(i)
    return {k: np.array(v) for k, v in sorted(g.items())}


def tile_batch(Hd, Wn, CMt, idx, tw, th):
    """yvvyee: nl_tile.Data.batch 와 같은 입력 (W 는 PIXN 정규화, bsz = (PW>>2, PH>>2), bi = st>0)."""
    g = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(DEV); h = Hd[idx]
    return dict(W=g(Wn[idx, :, :th + 16, :tw + 16].astype(np.float64)).to(DT) / PIXN, fr=g(h[:, 11:15].astype(np.float64)).to(DT), qp=g(h[:, 1].astype(np.float64)).to(DT),
                bi=g(h[:, 0] > 0), imv=g(h[:, 17]), merge=g(h[:, 18]), dpoc=g(h[:, 19:21]), bsz=g(np.stack([h[:, 5] >> 2, h[:, 6] >> 2], 1)),
                cm=g(CMt[idx, :, :th + 16, :tw + 16].astype(np.int32)), st=g(h[:, 0]), tw=tw, th=th)


def load_export_scales(net, d):
    """yvvyee: 내보내기 디렉터리의 층별 출력 스케일 지수를 Q.scale 에 넣는다 (C++ 커널과 같은 스케일). 가중치 int16 은 대조만 한다."""
    man = {}
    for ln in open(f'{d}/manifest.txt'):
        t = ln.split(); nd = int(t[2]); man[t[0]] = np.fromfile(f'{d}/{t[3 + nd]}', dtype=t[1]).reshape([int(v) for v in t[3:3 + nd]])
    h = net.h; Q.scale = {}; Q.calib = False
    for gname, seq in (('noise', h.noise), ('dq', h.dq), ('corr', h.corr)):
        for i, m in enumerate(seq):
            if isinstance(m, nn.Conv2d):
                Q.scale[id(m)] = 2.0 ** int(man[f'conv_{gname}_{i}_oexp'][0])
                e = int(man[f'conv_{gname}_{i}_wexp'][0]); wq = (Q.weight(m.weight.detach().double()) * 2 ** e).round().cpu().numpy()
                assert (wq == man[f'conv_{gname}_{i}_w']).all(), f'가중치 불일치 conv_{gname}_{i} (체크포인트와 내보내기가 다름)'
            elif m.__class__.__name__ == 'FiltAct':
                eh, ev = man[f'filtact_{gname}_{i}_oexp']; Q.scale[(id(m), 'h')] = 2.0 ** int(eh); Q.scale[(id(m), 'v')] = 2.0 ** int(ev)
    Q.scale['corr_in'] = 2.0 ** int(man['corr_in_exp'][0])
    for key in (('sq', 0), ('sq', 1), ('sq', 2), ('sq', 3), ('phi', 0), ('phi', 1), ('phi', 2), ('phi', 3), 'st0', 'st', 'phin'):
        nm = key if isinstance(key, str) else f'{key[0]}{key[1]}'; Q.scale[('e', key)] = 2.0 ** int(man[f'energy_{nm}_exp'][0])
    return man


def tile_eval(ck, files, export_dir, ref_out=None, bs=128):
    """yvvyee: (tw,th) 그룹별로 (i) 양자화 끔 모사 대 부동소수 모형, (ii) 고정소수 모사 대 부동소수 모형의 차(10비트 단위, 유효 타일 영역)를 내고,
    ref_out 이 있으면 레코드·가설별 고정소수 출력(Q10.4 정수: y th*tw, y_lin (th+4)*(tw+4), xd (th+16)*(tw+16), 최종 출력 th*tw)을 int32 로 쓴다.
    행 = [rec, hyp, tw, th] + y[256] + ylin[400] + xd[1024] + yfin[256] (조밀 행 우선, 나머지 0). 양방향 레코드는 가설 1 행도 쓴다."""
    sd = torch.load(ck, map_location='cpu'); net = QKVMC(no_l9=True, ch=ckpt_ch(sd)); net.load_state_dict(sd); net.eval(); net.to(DEV)
    if DT == torch.float64: net.double()
    load_export_scales(net, export_dir); nq = NetQ(net); hq = nq.hq
    Hd, Wn, CMt = load_tile(files); G = tile_groups(Hd)
    print('tile records', len(Hd), '| groups', {k: len(v) for k, v in G.items()}, '| strata', np.bincount(Hd[:, 0], minlength=3).tolist(), flush=True)
    rows = []; stats = {}
    with torch.no_grad():
        for (tw, th), gidx in G.items():
            a = stats.setdefault((tw, th), dict(n=0, id_max=0.0, id_sum=0.0, q_max=0.0, q_sum=0.0, npx=0))
            for s0 in range(0, len(gidx), bs):
                idx = gidx[s0:s0 + bs]; b = tile_batch(Hd, Wn, CMt, idx, tw, th)
                bf = dict(b); bf['W'] = b['W'] * (PIXN / 1023.0); Pf = (net(bf) * 1023).cpu().numpy()
                Q.on = False; Pid = (nq.forward(b) * PIXN).cpu().numpy(); Q.on = True
                Pq = (nq.forward(b) * PIXN).cpu().numpy()
                d_id = np.abs(Pid - Pf); d_q = np.abs(Pq - Pf)
                a['n'] += len(idx); a['npx'] += d_id.size; a['id_max'] = max(a['id_max'], float(d_id.max())); a['id_sum'] += float(d_id.sum()); a['q_max'] = max(a['q_max'], float(d_q.max())); a['q_sum'] += float(d_q.sum())
                if ref_out is None: continue
                fr = b['fr'].clamp(min=0); q4 = lambda t: (t * PIXN * 16).round().cpu().numpy().astype(np.int64)
                yfin = q4(nq.forward(b))
                for hyp in (0, 1):
                    sel = np.arange(len(idx)) if hyp == 0 else np.where(Hd[idx, 0] > 0)[0]
                    if len(sel) == 0: continue
                    ts = torch.from_numpy(sel).to(DEV)
                    p = hq.forward(b['W'][ts, hyp], fr[ts, 2 * hyp:2 * hyp + 2], b['qp'][ts], b['cm'][ts, hyp], b['imv'][ts], b['merge'][ts], b['dpoc'][ts, hyp], b['bsz'][ts], b['st'][ts], tw=tw, th=th)
                    y, yl, xd = q4(p['y']), q4(p['y_lin']), q4(p['xd'])
                    R = np.zeros((len(sel), 4 + 256 + 400 + 1024 + 256), np.int64)
                    R[:, 0] = idx[sel]; R[:, 1] = hyp; R[:, 2] = tw; R[:, 3] = th
                    R[:, 4:4 + th * tw] = y.reshape(len(sel), -1); R[:, 260:260 + (th + 4) * (tw + 4)] = yl.reshape(len(sel), -1)
                    R[:, 660:660 + (th + 16) * (tw + 16)] = xd.reshape(len(sel), -1); R[:, 1684:1684 + th * tw] = yfin[sel].reshape(len(sel), -1)
                    rows.append(R)
            print(f'group {tw}x{th} done ({a["n"]})', flush=True)
    print(f'{"tile":>6s} {"n":>6s} | {"sim(off)-float max":>18s} {"mean":>9s} | {"fixed-float max":>15s} {"mean":>8s}   (10비트 단위, 출력 화소)')
    for (tw, th), a in stats.items():
        print(f'{tw:>2d}x{th:<3d} {a["n"]:6d} | {a["id_max"]:18.4f} {a["id_sum"] / a["npx"]:9.5f} | {a["q_max"]:15.3f} {a["q_sum"] / a["npx"]:8.4f}')
    if ref_out is not None:
        R = np.concatenate(rows); assert np.abs(R).max() < 2 ** 31; R.astype(np.int32).tofile(ref_out); print('ref rows', len(R), '->', ref_out)

if __name__ == '__main__':
    torch.set_num_threads(8)
    if sys.argv[1] == '--tile':   # yvvyee: python qkv_fixed.py --tile ckpt.pt tile.dmp export_dir [ref_out.bin] [--f64]
        args = [a for a in sys.argv[2:] if not a.startswith('--')]; tile_eval(args[0], [args[1]], args[2], args[3] if len(args) > 3 else None); sys.exit(0)
    if sys.argv[1] == '--stream':
        args = [a for a in sys.argv[2:] if not a.startswith('--')]; ck = args[0]; files = sorted(glob.glob(args[1])); stream_eval(ck, files, files); sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; ck = args[0]; files = args[1:]
    sd = torch.load(ck, map_location='cpu')
    net = QKVMC(no_l9=('--no-l9' in sys.argv), ch=ckpt_ch(sd)); net.load_state_dict(sd); net.eval(); net.to(DEV)
    if DT == torch.float64: net.double()
    nq = NetQ(net)
    M, W, T, C0, H2, CM, st, bi = load_dump(files)
    print('records', len(M), 'strata', np.bincount(st, minlength=3))
    Pf = np.zeros((len(M), 4, 4), np.float32); Pq = np.zeros_like(Pf); Pid = np.zeros_like(Pf)
    with torch.no_grad():
        Q.calib = True
        for j, b in batches(M, W, H2, CM, st, bi):
            nq.forward(b)
            if j.stop >= 4096: break
        Q.calib = False
        print('보정된 층 수', len(Q.amax))
        for j, b in batches(M, W, H2, CM, st, bi):
            bf = dict(b); bf['W'] = b['W'] * (PIXN / 1023.0); Pf[j] = (net(bf) * 1023).cpu().numpy()
            Q.on = False; Pid[j] = (nq.forward(b) * PIXN).cpu().numpy(); Q.on = True
            Pq[j] = (nq.forward(b) * PIXN).cpu().numpy()
        Pg = {}
        for gname in ('act', 'lut'):
            Q.groups = {gname}; Pg[gname] = np.zeros_like(Pf)
            for j, b in batches(M, W, H2, CM, st, bi): Pg[gname][j] = (nq.forward(b) * PIXN).cpu().numpy()
        Q.groups = set(Quant.GROUPS)
    d_id = np.abs(Pid - Pf); d_q = np.abs(Pq - Pf)
    print(f'모사 경로(양자화 끔) 대 원 모형: 최대 차 {d_id.max():.4f}, 평균 {d_id.mean():.5f} (10비트 단위)  <- 0 에 가까워야 경로가 같음')
    print(f'고정소수 모사 대 부동소수: 최대 차 {d_q.max():.3f}, 평균 {d_q.mean():.4f}, 차 > 0.5 인 화소 비율 {(d_q > 0.5).mean():.4f}')
    def rel(m, X): e0 = ((T[m] - C0[m]) ** 2).sum(); return 100 * (((T[m] - X[m]) ** 2).sum() / e0 - 1)
    for s_, nm in ((0, 'uni'), (1, 'bi noBDOF'), (2, 'bi BDOF')):
        m = st == s_
        if m.sum(): print(f'  {nm:10s} n={m.sum():7d} | float {rel(m, Pf):+.3f} | fixed {rel(m, Pq):+.3f} | 반올림(화소 정수) float {rel(m, np.round(Pf)):+.3f} fixed {rel(m, np.round(Pq)):+.3f}')
    print('양자화 그룹 하나만 켰을 때 (최대 차 / 평균 차 / uni / BDOF SSE %):')
    for gname in ('act', 'lut'):
        d = np.abs(Pg[gname] - Pf); print(f'  {gname:7s} {d.max():8.3f} {d.mean():8.4f} | uni {rel(st == 0, Pg[gname]):+.3f} | BDOF {rel(st == 2, Pg[gname]):+.3f}')
    ov = sorted(Q.overflow.values(), reverse=True)[:3]
    print('합성곱 pre-activation 최대 |값| 상위 3 (Q8.8 가정 시 128 넘으면 포화):', [round(v, 2) for v in ov])
