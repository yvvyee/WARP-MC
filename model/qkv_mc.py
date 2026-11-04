"""yvvyee: 3단계 개정 설계(stage3_design.md 8~10절) — 이론 대응 쿼리·키·밸류 보간기 QKV-MC.

DCTIF 계수는 쓰지 않는다. 입력은 가설별 20x20 정수 창(좌상단 = 정수 MV 위치 - 8), 가설별 참조 부호화 구조 지도(10.2절 비트 배치),
MV 분수부(1/16), 슬라이스 QP, AMVR 등급(imv), 병합 플래그, 가설별 참조 POC 거리, 블록 크기다. 출력은 창 인덱스 8..11 의 4x4 예측이다.

1단계 이론의 최적 필터를 층으로 옮긴다.

    F*(w; s) = e^{jws} * P*(w) * Phi(w) / (Phi(w) + Phi_nn(w))      (단방향, 접힘 무시 항)

  L1 대역 분해      B_b               : 주파수 축을 유한 탭 공간 필터로 나눈다 (합 = 항등)
  L2 키 (내용)      Phi_b(x)          : 참조 창의 국소 대역 에너지
  L3 쿼리 (잡음)    Phi_nn,b(x)       : QP 와 참조 부호화 구조(TU 경계 거리, bS, cbf, 인트라, 참조 CU QP)로 예측한 잡음 대역 에너지
  L4 Wiener 가중    G_b = Phi/(Phi+Phi_nn)
  L4b AMVR 반화소   축별 대칭 7탭 학습 저역통과 (초기 항등). imv=3 이고 그 축의 MV 분수가 8일 때만 적용 (전환식 반화소 필터 구문 구조)
  L5 밸류           V_delta           : 후보 변위 s + Delta 로 이동한 표본 (창 sinc, 학습 없음)
  L6 변위 주의      alpha_delta(x)    : softmax_delta <q(x), phi(Delta)> -> sum alpha V = e^{jws} P*(w)
                                        q 는 QP, AMVR, 병합, POC 거리, 블록 크기, 구조 텐서, 고주파 잡음 추정으로 만든다
  L7 비선형 보정    r(x)              : 대역 제한 비선형 CNN (단방향 접힘 몫), 정렬 창 + 잡음·경계 특징 입력
  L8 율 문턱        c_r 문턱          : 양자화 스텝 아래 보정 성분 억제
  L9 양방향 결합    beta_b(x)         : 위상 차, 가설별 잡음·신호, POC 거리, 층으로 대역별 결합 (P4)
                                        BDOF 미적용 층과 적용 층의 결합 헤드를 분리하고, BDOF 조건(반대 방향·같은 거리·크기)을
                                        충족하는데 미적용인 블록(SMVD 류) 파생 플래그를 조건에 넣는다 (13.8절 진단)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WS = 20          # 정수 창 크기
C0 = 8           # 창에서 출력 4x4 의 시작 인덱스
NCM = 12         # 부호화 구조 지도에서 뽑는 화소별 특징 수


# ---------------------------------------------------------------- 고정 연산
def gauss1d(sigma, r):
    t = torch.arange(-r, r + 1, dtype=torch.float64)
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    return (g / g.sum()).float()


def sep_blur(x, h):
    """채널별 분리형 저역통과. 복제 패딩으로 창 경계 밖을 채운다."""
    c = x.shape[1]; r = (len(h) - 1) // 2
    kx = h.view(1, 1, 1, -1).repeat(c, 1, 1, 1).to(x); ky = h.view(1, 1, -1, 1).repeat(c, 1, 1, 1).to(x)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), kx, groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), ky, groups=c)


def kaiser_frac(f, taps=12, beta=6.0):
    """분수 위치 f (임의 형상, 0<=f<1)에서의 창 sinc 탭. 반환 (..., taps), 탭 k 는 정수 오프셋 k-(taps/2-1)."""
    k = torch.arange(taps, dtype=f.dtype, device=f.device) - (taps // 2 - 1)
    t = k - f[..., None]
    R = taps / 2
    win = torch.special.i0(beta * torch.sqrt((1 - (t / R) ** 2).clamp(min=0))) / torch.special.i0(torch.tensor(beta, dtype=f.dtype))
    h = torch.sinc(t) * win
    return h / h.sum(-1, keepdim=True)


def shift_crop(X, sx, sy, r0, n, taps=12):
    """X (N,H,W) 를 (sx, sy) 화소 이동한 뒤 [r0, r0+n) 영역을 잘라 (N,n,n) 반환.
    출력[y,x] = X(r0+y+sy, r0+x+sx). 정수부는 인덱스로, 분수부는 창 sinc 로 처리한다."""
    N, H, W = X.shape
    dev = X.device
    base = torch.arange(n, device=dev, dtype=X.dtype) + r0
    kk = torch.arange(taps, device=dev) - (taps // 2 - 1)

    def axis_idx_w(s, L):
        pos = base[None] + s[:, None]                       # (N,n)
        i0 = torch.floor(pos); f = pos - i0
        idx = (i0.long()[..., None] + kk).clamp(0, L - 1)   # (N,n,taps) 창 밖은 가장자리 복제
        return idx, kaiser_frac(f, taps)
    ix, wx = axis_idx_w(sx, W); iy, wy = axis_idx_w(sy, H)
    g = torch.gather(X[:, :, None, :].expand(N, H, n, W), 3, ix[:, None].expand(N, H, n, taps))
    Xh = (g * wx[:, None]).sum(-1)                           # (N,H,n)
    g = torch.gather(Xh[:, None].expand(N, n, H, n), 2, iy[..., None].expand(N, n, taps, n))
    return (g * wy[..., None]).sum(2)                        # (N,n,n)


def shift_crop2(X, sx, sy, r0, ny, nx, taps=12):
    """shift_crop 의 직사각형판: 영역 [r0, r0+ny) x [r0, r0+nx)."""
    N, H, W = X.shape; dev = X.device; kk = torch.arange(taps, device=dev) - (taps // 2 - 1)
    def axis(s, L, n):
        pos = torch.arange(n, device=dev, dtype=X.dtype)[None] + r0 + s[:, None]; i0 = torch.floor(pos); f = pos - i0
        return (i0.long()[..., None] + kk).clamp(0, L - 1), kaiser_frac(f, taps)
    ix, wx = axis(sx, W, nx); iy, wy = axis(sy, H, ny)
    g = torch.gather(X[:, :, None, :].expand(N, H, nx, W), 3, ix[:, None].expand(N, H, nx, taps)); Xh = (g * wx[:, None]).sum(-1)      # (N,H,nx)
    g = torch.gather(Xh[:, None].expand(N, ny, H, nx), 2, iy[..., None].expand(N, ny, taps, nx)); return (g * wy[..., None]).sum(2)   # (N,ny,nx)


class FiltAct(nn.Module):
    """대역 제한 비선형 (P1): 2배 업샘플(창 sinc) -> LeakyReLU -> 저역통과 -> 2배 다운샘플.
    점별 비선형이 만든 나이퀴스트 이상 성분이 접혀 들어오지 않게 한다."""
    def __init__(self, taps=6):
        super().__init__()
        n = torch.arange(2 * taps, dtype=torch.float64) - (2 * taps - 1) / 2
        h = 0.5 * torch.sinc(0.5 * n) * torch.kaiser_window(2 * taps, periodic=False, beta=8.0, dtype=torch.float64)
        h = (h / h.sum()).float()
        self.register_buffer('hu', h * 2); self.register_buffer('hd', h)

    @staticmethod
    def sep(x, h):
        c = x.shape[1]; t = len(h)
        kx = h.view(1, 1, 1, t).repeat(c, 1, 1, 1).to(x); ky = h.view(1, 1, t, 1).repeat(c, 1, 1, 1).to(x)
        x = F.conv2d(F.pad(x, (t // 2, t // 2 - 1, 0, 0), mode='reflect'), kx, groups=c)
        return F.conv2d(F.pad(x, (0, 0, t // 2, t // 2 - 1), mode='reflect'), ky, groups=c)

    def forward(self, x):
        b, c, h, w = x.shape
        u = x.new_zeros(b, c, 2 * h, 2 * w); u[:, :, ::2, ::2] = x
        u = F.leaky_relu(self.sep(u, self.hu), 0.2)
        return self.sep(u, self.hd)[:, :, ::2, ::2]


def precision_onehot(fr):
    """MV 분수부로 본 실제 보간 정밀도: 정수 / 반화소 / 1/4 / 1/16."""
    fi = fr.long()
    integ = (fi == 0).all(1); half = ((fi % 8) == 0).all(1) & ~integ; quar = ((fi % 4) == 0).all(1) & ~integ & ~half
    return torch.stack([integ, half, quar, ~(integ | half | quar)], 1).float()


def _scan_dist(flag, val, dim):
    """경계 플래그(1 = 이 화소와 이전 화소 사이가 경계)에서, 각 화소의 뒤쪽(작은 좌표) 가장 가까운 경계까지 거리와 그 경계의 값,
    앞쪽(큰 좌표) 가장 가까운 경계까지 거리와 값을 구한다. 경계가 없으면 거리 n, 값 0. 반환 4개 텐서 (flag 와 같은 형상)."""
    dim = dim % flag.dim(); n = flag.shape[dim]
    shape = [1] * flag.dim(); shape[dim] = -1
    idx = torch.arange(n, device=flag.device).view(shape).expand_as(flag)
    neg = torch.full_like(idx, -1); big = torch.full_like(idx, n + 1)
    last = torch.cummax(torch.where(flag > 0, idx, neg), dim).values                   # 플래그가 있는 e <= x 중 최대
    d_back = torch.where(last >= 0, idx - last, torch.full_like(idx, n))
    v_back = torch.where(last >= 0, torch.gather(val, dim, last.clamp(min=0)), torch.zeros_like(val))
    fl = torch.where(flag > 0, idx, big)
    fl1 = torch.cat([fl.narrow(dim, 1, n - 1), big.narrow(dim, 0, 1)], dim)             # 위치 x 에 x+1 의 값
    nxt = torch.flip(torch.cummin(torch.flip(fl1, [dim]), dim).values, [dim])            # 플래그가 있는 e >= x+1 중 최소
    d_fwd = torch.where(nxt <= n, nxt - 1 - idx, torch.full_like(idx, n))               # 경계는 e-1 과 e 사이
    v_fwd = torch.where(nxt <= n, torch.gather(val, dim, nxt.clamp(max=n - 1)), torch.zeros_like(val))
    return d_back, v_back, d_fwd, v_fwd


def codemap_feats(cm, qp_slice):
    """부호화 구조 지도 (N,h,w) int -> 화소별 특징 (N,NCM,h,w).
    [0..3] 왼쪽·오른쪽·위쪽·아래쪽 가장 가까운 TU 경계까지 거리 (8 화소에서 절단, /8)
    [4..7] 그 경계들의 휘도 bS (/2)   [8] cbf   [9] 인트라   [10] (참조 CU QP - 슬라이스 QP)/12   [11] 지도 유효"""
    cm = cm.long()
    eL = (cm & 1); eT = (cm >> 1) & 1; bL = (cm >> 2) & 3; bT = (cm >> 4) & 3
    cbf = (cm >> 6) & 1; intra = (cm >> 7) & 1; qpr = (cm >> 8) & 63; valid = (cm >> 14) & 1
    dl, bl, dr, br = _scan_dist(eL, bL, -1)
    dt, bt, db, bb = _scan_dist(eT, bT, -2)
    f = lambda d: d.clamp(max=8).float() / 8
    dqp = (qpr.float() - qp_slice.view(-1, 1, 1)) / 12 * valid.float()
    return torch.stack([f(dl), f(dr), f(dt), f(db), bl.float() / 2, br.float() / 2, bt.float() / 2, bb.float() / 2,
                        cbf.float(), intra.float(), dqp, valid.float()], 1)


# ---------------------------------------------------------------- 단방향 가설 보간
class Hypothesis(nn.Module):
    SIG = (0.6, 1.2, 2.4)            # 대역 경계 (가우시안 표준편차, 화소)

    def __init__(self, ch=32, depth=5, nd=2, step=2.0, rate_gate=True, filt=True):
        super().__init__()
        act = (lambda: FiltAct()) if filt else (lambda: nn.LeakyReLU(0.2))   # P1 절제: 대역 제한 비선형 대신 점별 LeakyReLU
        d = torch.arange(-nd, nd + 1, dtype=torch.float32) * step / 16            # 후보 오프셋 Delta: (2nd+1)^2 개
        dy, dx = torch.meshgrid(d, d, indexing='ij')
        self.register_buffer('dx', dx.reshape(-1)); self.register_buffer('dy', dy.reshape(-1))
        D = torch.stack([dx * dx, dx * dy, dy * dy, dx, dy], -1).reshape(-1, 5) * torch.tensor([256., 256., 256., 16., 16.])
        self.register_buffer('phi', D)                                             # 키 phi(Delta): 가우시안 로그 밀도 기저
        self.nb = len(self.SIG) + 1
        self.nside = 1 + 4 + 4 + 1 + 1 + 2 + 3                                     # QP, imv, 정밀도, 병합, |dPOC|, log2 블록 크기, 층 one-hot
        # L4b: 축별 대칭 7탭 학습 커널 [h3,h2,h1,h0,h1,h2,h3]/합 (초기 항등)과 적용 게이트 (초기 sigmoid(2)=0.88)
        self.alt_k = nn.Parameter(torch.tensor([0., 0., 0., 1.])); self.alt_g = nn.Parameter(torch.tensor(2.0))
        # L3 잡음 쿼리: 부호화 조건 + 지도 특징 -> 대역별 log Phi_nn (위치별)
        self.noise = nn.Sequential(nn.Conv2d(self.nside + NCM, 32, 1), nn.ReLU(), nn.Conv2d(32, self.nb, 1))
        # L6 변위 쿼리: 부호화 조건 + 구조 텐서 + 고주파 잡음 추정 + 경계 거리 -> 5차원 쿼리 (위치별)
        self.dq = nn.Sequential(nn.Conv2d(self.nside + 3 + 1 + 4, 32, 1), nn.ReLU(), nn.Conv2d(32, 5, 1))
        nn.init.zeros_(self.dq[-1].weight); nn.init.zeros_(self.dq[-1].bias)
        with torch.no_grad():                    # 초기: 날카로운 가우시안 (Delta=0 에 집중)
            self.dq[-1].bias[:] = torch.tensor([-2.0, 0.0, -2.0, 0.0, 0.0])
        # L7 비선형 보정 (대역 제한 CNN): 정렬 창 + 잡음 추정 4대역 + 경계 특징 6
        L = [nn.Conv2d(1 + self.nb + 6, ch, 3, padding=1, padding_mode='reflect'), act()]
        for _ in range(depth - 2): L += [nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect'), act()]
        L += [nn.Conv2d(ch, 1, 3, padding=1, padding_mode='reflect')]
        self.corr = nn.Sequential(*L)
        nn.init.zeros_(self.corr[-1].weight); nn.init.zeros_(self.corr[-1].bias)
        # L8 율 문턱 (양자화 스텝 배수, 학습)
        self.rate_gate = rate_gate
        self.tau = nn.Parameter(torch.tensor(-2.0))
        D4 = torch.tensor([[math.cos(math.pi * (2 * n + 1) * k / 8) * (0.5 if k == 0 else math.sqrt(0.5)) for n in range(4)] for k in range(4)])
        self.register_buffer('D4', D4)

    # L1 대역 분해: B0 = G2.4, B1 = G1.2-G2.4, B2 = G0.6-G1.2, B3 = I-G0.6 (합이 정확히 항등)
    def bands(self, x):
        g = [sep_blur(x, gauss1d(s, int(math.ceil(3 * s)))) for s in self.SIG]
        return [g[2], g[1] - g[2], g[0] - g[1], x - g[0]]

    def side(self, qp, fr, imv, merge, dpoc, bsz, st, n):
        s = torch.cat([qp[:, None] / 63.0, F.one_hot(imv.long().clamp(0, 3), 4).float(), precision_onehot(fr),
                       merge.float()[:, None], dpoc.abs().float()[:, None] / 16.0, torch.log2(bsz.float().clamp(min=1))[:, :2] / 5.0,
                       F.one_hot(st.long().clamp(0, 2), 3).float()], 1)
        return s[:, :, None, None].expand(-1, -1, n, n)

    def forward(self, W, fr, qp, cm, imv, merge, dpoc, bsz, st, return_parts=False, tw=4, th=4):
        """타일 의미(stage4 11절): 창 (th+16)x(tw+16), 출력 tw x th. tw=th=4 이면 20x20 창의 블록 의미와 같다."""
        N = W.shape[0]; Hh, Ww = W.shape[-2], W.shape[-1]; assert Hh == th + 16 and Ww == tw + 16
        x = W[:, None]                                             # (N,1,H,W)
        sd = self.side(qp, fr, imv, merge, dpoc, bsz, st, Hh)[..., :Hh, :Ww] if Hh == Ww else self.side(qp, fr, imv, merge, dpoc, bsz, st, max(Hh, Ww))[..., :Hh, :Ww]
        cf = codemap_feats(cm, qp)                                 # (N,NCM,20,20)
        # L1·L2 키: 대역 분해와 국소 대역 에너지 (5x5 평균)
        B = self.bands(x)
        phi_s = torch.cat([sep_blur(b * b, gauss1d(1.0, 2)) for b in B], 1)            # (N,nb,20,20)
        # L3 쿼리: 부호화 잡음 대역 에너지 (초기 e^-30 -> G~1, 항등에서 출발)
        phi_n = torch.exp(self.noise(torch.cat([sd, cf], 1)) - 30.0)
        # L4 Wiener 가중
        G = phi_s / (phi_s + phi_n + 1e-12)
        xd = sum(G[:, b:b + 1] * B[b] for b in range(self.nb))[:, 0]                  # (N,20,20)
        # L4b AMVR 반화소 저역통과 (축별, 해당 축 분수 8 인 imv=3 블록에만)
        k = torch.cat([self.alt_k, self.alt_k[:3].flip(0)]); k = (k / k.sum()).to(xd.dtype)
        ax = ((imv == 3) & (fr[:, 0] == 8)).to(xd.dtype).view(-1, 1, 1) * torch.sigmoid(self.alt_g)
        ay = ((imv == 3) & (fr[:, 1] == 8)).to(xd.dtype).view(-1, 1, 1) * torch.sigmoid(self.alt_g)
        lx = F.conv2d(F.pad(xd[:, None], (3, 3, 0, 0), mode='replicate'), k.view(1, 1, 1, 7))[:, 0]
        xd = xd + ax * (lx - xd)
        ly = F.conv2d(F.pad(xd[:, None], (0, 0, 3, 3), mode='replicate'), k.view(1, 1, 7, 1))[:, 0]
        xd = xd + ay * (ly - xd)
        # L6 쿼리: 구조 텐서 (조리개 문제) + 고주파 잡음 추정 + TU 경계 거리
        gx = F.pad(xd[:, :, 2:] - xd[:, :, :-2], (1, 1, 0, 0), mode='replicate') / 2
        gy = F.pad(xd[:, 2:] - xd[:, :-2], (0, 0, 1, 1), mode='replicate') / 2
        st = sep_blur(torch.stack([gx * gx, gx * gy, gy * gy], 1), gauss1d(1.0, 2)) * 1023.0
        lpn = torch.log(phi_n + 1e-12) / 30.0
        q = self.dq(torch.cat([sd, torch.log1p(st.abs()) * st.sign(), lpn[:, 3:4], cf[:, 0:4]], 1))   # (N,5,20,20)
        # L5 밸류: 후보 변위 s + Delta 로 이동한 8x8 영역 (창 인덱스 6..13)
        s_x = fr[:, 0] / 16; s_y = fr[:, 1] / 16
        K = len(self.dx)
        Xr = xd[:, None].expand(N, K, Hh, Ww).reshape(N * K, Hh, Ww)
        V = shift_crop2(Xr, (s_x[:, None] + self.dx[None]).reshape(-1), (s_y[:, None] + self.dy[None]).reshape(-1), 6, th + 4, tw + 4).view(N, K, th + 4, tw + 4)
        # L6 변위 주의: logit = <q(x), phi(Delta)>, 위치마다 후보에 대해 softmax
        qc = q[:, :, 6:6 + th + 4, 6:6 + tw + 4]
        alpha = torch.softmax(torch.einsum('nchw,kc->nkhw', qc, self.phi), 1)
        y_lin = (alpha * V).sum(1)                                                       # (N,th+4,tw+4)
        # L7 비선형 보정: s 로 정렬한 (타일+12) 문맥 + 잡음 추정 + 경계 특징 (지도는 정수 격자 그대로, 오차 < 1 화소)
        A = shift_crop2(xd, s_x, s_y, 2, th + 12, tw + 12)
        r = self.corr(torch.cat([A[:, None], lpn[:, :, 2:2 + th + 12, 2:2 + tw + 12], cf[:, [0, 1, 2, 3, 8, 9], 2:2 + th + 12, 2:2 + tw + 12]], 1))[:, 0, 6:6 + th, 6:6 + tw]
        # L8 율 문턱: 보정의 4x4 DCT 계수 중 양자화 스텝의 exp(tau) 배 아래 성분을 부드럽게 억제
        if self.rate_gate:
            step = (4 * 2 ** ((qp - 4) / 6) / 1023.0).view(-1, 1, 1)
            rb = r.reshape(N, th // 4, 4, tw // 4, 4).permute(0, 1, 3, 2, 4)                # (N,bh,bw,4,4) 4x4 블록별
            c = self.D4 @ rb @ self.D4.T
            thr = (torch.exp(self.tau) * step).view(N, 1, 1, 1, 1)
            c = c * torch.sigmoid((c.abs() - thr) / (0.25 * thr))
            r = (self.D4.T @ c @ self.D4).permute(0, 1, 3, 2, 4).reshape(N, th, tw)
        y = y_lin[:, 2:2 + th, 2:2 + tw] + r
        if return_parts:
            return dict(y=y, y_lin=y_lin, alpha=alpha, G=G, xd=xd, phi_n=phi_n, phi_s=phi_s, B=B, cf=cf)
        return y


# ---------------------------------------------------------------- 양방향 결합 (P4)
class QKVMC(nn.Module):
    def __init__(self, pd_gate=False, no_l9=False, **kw):
        super().__init__()
        self.pd_gate = pd_gate      # L9 평균 이탈 폭을 위상 차로 묶음: g = 1-(1-|sin pi dphi_x|)(1-|sin pi dphi_y|) (1단계 식 7)
        self.no_l9 = no_l9          # L9 제거: 양방향 = 가설별 출력 평균 (BDOF 층 선택지 2 용 가설별 모형)
        filt = kw.get('filt', True)
        self.h = Hypothesis(**kw)
        nb = self.h.nb
        # L9: [sin/cos(2 pi (s0-s1)) 4, QP, dPOC0, dPOC1, 층 one-hot 2, SMVD류 플래그, 가설별 log Phi_s, log Phi_nn (대역별)] -> 대역별 결합 logit
        #     헤드 0 = BDOF 미적용 층, 헤드 1 = BDOF 적용 층 (초기 모두 평균)
        def comb_head():
            m = nn.Sequential(nn.Conv2d(10 + 4 * nb, 32, 1), nn.ReLU(), nn.Conv2d(32, nb, 1)); nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        def bcorr_head():   # 양방향 차분 보정: 정렬된 두 가설의 차는 접힘 성분 정보를 담는다 (1단계 식 7)
            m = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, padding_mode='reflect'), FiltAct() if filt else nn.LeakyReLU(0.2), nn.Conv2d(16, 1, 3, padding=1, padding_mode='reflect'))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        self.comb = nn.ModuleList([comb_head(), comb_head()]); self.bcorr = nn.ModuleList([bcorr_head(), bcorr_head()])

    def forward(self, b):
        W, fr, qp, bi, cm = b['W'], b['fr'].clamp(min=0), b['qp'], b['bi'], b['cm']
        imv, merge, dpoc, bsz, st = b['imv'], b['merge'], b['dpoc'], b['bsz'], b['st']
        tw, th = b.get('tw', 4), b.get('th', 4)
        if (tw, th) != (4, 4): assert self.no_l9, 'L9 경로는 4x4 블록 의미에서만 구현됨'
        p0 = self.h(W[:, 0], fr[:, 0:2], qp, cm[:, 0], imv, merge, dpoc[:, 0], bsz, st, return_parts=True, tw=tw, th=th)
        if not bool(bi.any()):
            return p0['y']
        ib = bi.nonzero()[:, 0]
        p1 = self.h(W[ib, 1], fr[ib, 2:4], qp[ib], cm[ib, 1], imv[ib], merge[ib], dpoc[ib, 1], bsz[ib], st[ib], return_parts=True, tw=tw, th=th)
        if self.no_l9:
            out = p0['y'].clone(); out[ib] = 0.5 * (p0['y'][ib] + p1['y']); return out
        y0 = p0['y_lin'][ib]; y1 = p1['y_lin']
        nb = self.h.nb
        B0 = self.h.bands(y0[:, None]); B1 = self.h.bands(y1[:, None])
        dphi = (fr[ib, 0:2] - fr[ib, 2:4]) * (2 * math.pi / 16)
        d0, d1 = dpoc[ib, 0].long(), dpoc[ib, 1].long(); w4, h4 = bsz[ib, 0].long(), bsz[ib, 1].long()
        elig = ((d0 * d1 < 0) & (d0.abs() == d1.abs()) & (w4 * h4 * 16 >= 128) & (torch.minimum(w4, h4) * 4 >= 8)).float()   # BDOF 기하 조건 충족
        smvd_like = (elig * (st[ib] == 1).float())[:, None]                                                                  # 충족인데 미적용 (SMVD 류)
        cond = torch.cat([torch.sin(dphi), torch.cos(dphi), qp[ib, None] / 63.0, dpoc[ib].float() / 16.0, F.one_hot(st[ib].long().clamp(0, 2), 3).float()[:, 1:], smvd_like], 1)[:, :, None, None].expand(-1, -1, 8, 8)
        stats = torch.cat([torch.log(p0['phi_s'][ib][:, :, 6:14, 6:14] + 1e-12), torch.log(p1['phi_s'][:, :, 6:14, 6:14] + 1e-12),
                           torch.log(p0['phi_n'][ib][:, :, 6:14, 6:14]), torch.log(p1['phi_n'][:, :, 6:14, 6:14])], 1) / 30.0
        xin = torch.cat([cond, stats], 1); sel = (st[ib] == 2).float().view(-1, 1, 1, 1)               # 층별 헤드 선택
        if self.pd_gate:
            sx = torch.sin(dphi[:, 0] / 2).abs(); sy = torch.sin(dphi[:, 1] / 2).abs()                  # |sin(pi dphi)| (dphi 는 2 pi 단위)
            gpd = (1 - (1 - sx) * (1 - sy)).view(-1, 1, 1, 1)
        else:
            gpd = torch.ones(len(ib), 1, 1, 1, device=W.device, dtype=W.dtype)
        beta = 0.5 + (torch.sigmoid(sel * self.comb[1](xin) + (1 - sel) * self.comb[0](xin)) - 0.5) * gpd   # (Nb,nb,8,8), 초기 0.5
        yb = sum(beta[:, k:k + 1] * B0[k] + (1 - beta[:, k:k + 1]) * B1[k] for k in range(nb))[:, 0]
        din = torch.stack([y0, y1, y0 - y1], 1)
        rb = (sel * self.bcorr[1](din) + (1 - sel) * self.bcorr[0](din))[:, 0, 2:6, 2:6] * gpd[:, 0]
        r0 = p0['y'][ib] - p0['y_lin'][ib][:, 2:6, 2:6]; r1 = p1['y'] - y1[:, 2:6, 2:6]
        out = p0['y'].clone()
        out[ib] = yb[:, 2:6, 2:6] + 0.5 * (r0 + r1) + rb
        return out


# ---------------------------------------------------------------- 단위 시험
def fake_codemap(N, g, qp_ref=30, p_edge=0.25, p_intra=0.15, p_cbf=0.3):
    """4화소 격자 위에 무작위 TU 경계·bS·cbf·인트라·QP 를 놓은 지도 (N,20,20)."""
    u = torch.zeros(N, 5, 5, dtype=torch.long)                       # 4x4 단위 (창 20 = 5 단위)
    eL = (torch.rand(N, 5, 5, generator=g) < p_edge).long(); eT = (torch.rand(N, 5, 5, generator=g) < p_edge).long()
    intra = (torch.rand(N, 5, 5, generator=g) < p_intra).long(); cbf = (torch.rand(N, 5, 5, generator=g) < p_cbf).long()
    bL = eL * torch.where(intra > 0, 2, cbf); bT = eT * torch.where(intra > 0, 2, cbf)
    u = eL | (eT << 1) | (bL << 2) | (bT << 4) | (cbf << 6) | (intra << 7) | (qp_ref << 8) | (1 << 14)
    px = u.repeat_interleave(4, 1).repeat_interleave(4, 2)             # 단위 -> 화소
    col = torch.arange(20) % 4 == 0; row = (torch.arange(20) % 4 == 0)
    px = torch.where(col[None, None, :], px, px & ~0x000D)             # 좌측 경계 정보는 단위 첫 열에서만
    px = torch.where(row[None, :, None], px, px & ~0x0032)             # 상단 경계 정보는 단위 첫 행에서만
    return px.int()


if __name__ == '__main__':
    g = torch.Generator().manual_seed(0); torch.manual_seed(0)
    N = 64
    base = torch.randn(N, 1, 10, 10, generator=g)
    Wt = F.interpolate(base, size=(WS, WS), mode='bicubic', align_corners=False)[:, 0]
    Wt = sep_blur(Wt[:, None], gauss1d(0.7, 3))[:, 0]
    Wt = (Wt - Wt.min()) / (Wt.max() - Wt.min())
    m = Hypothesis()
    # T1 대역 분해 합 = 항등
    B = m.bands(Wt[:, None])
    print('T1 대역 합 대 입력 최대 차:', float((sum(B) - Wt[:, None]).abs().max()))
    # T2 shift_crop: 정수 이동 = 인덱스 이동, 0.5+0.5 = 1
    s1 = shift_crop(Wt, torch.ones(N), torch.zeros(N), 6, 8)
    print('T2 정수 1화소 이동 최대 차:', float((s1 - Wt[:, 6:14, 7:15]).abs().max()))
    h = shift_crop(Wt, torch.full((N,), .5), torch.zeros(N), 2, 16)
    hh = shift_crop(h, torch.full((N,), .5), torch.zeros(N), 4, 8)
    print('T2 0.5+0.5 대 1화소 최대 차 (대역 제한 창):', float((hh - s1).abs().max()))
    # T3 변위 주의 = 특성 함수
    d = m.dx.unique(); w = torch.linspace(0, math.pi, 7)
    Pana = torch.stack([torch.exp(1j * wi * d).mean() for wi in w]).real
    resp = []
    for wi in w:
        x = torch.cos(wi * torch.arange(40.)).repeat(40, 1)[None]
        ys = torch.stack([shift_crop(x, torch.tensor([float(di)]), torch.zeros(1), 16, 8) for di in d]).mean(0)
        y0 = shift_crop(x, torch.zeros(1), torch.zeros(1), 16, 8)
        resp.append(float((ys * y0).sum() / (y0 * y0).sum()))
    print('T3 균등 후보 가중의 응답 vs 해석 P(w):', ' '.join(f'{a:.4f}/{b:.4f}' for a, b in zip(resp, Pana.tolist())))
    # T7 지도 특징: 손으로 만든 행에서 경계 거리와 bS 확인
    row = torch.zeros(1, 1, 20, dtype=torch.long)
    row[0, 0, 4] = 1 | (2 << 2); row[0, 0, 12] = 1 | (1 << 2)      # x=4 왼쪽에 bS2 경계, x=12 왼쪽에 bS1 경계
    cf = codemap_feats((row | (1 << 14)).int(), torch.tensor([32.]))
    dl, dr, bl, br = cf[0, 0, 0] * 8, cf[0, 1, 0] * 8, cf[0, 4, 0] * 2, cf[0, 5, 0] * 2
    print('T7 x=6: 왼쪽 거리/bS', float(dl[6]), float(bl[6]), '(기대 2, 2) | 오른쪽 거리/bS', float(dr[6]), float(br[6]), '(기대 5, 1)',
          '| x=0 왼쪽 거리', float(dl[0]), '(기대 8=절단) | x=15 오른쪽 거리', float(dr[15]), '(기대 8=절단)')
    # T4 초기 상태: G~1, alpha 가 Delta=0 에 집중, 보정 0 -> 출력 = 창 sinc 이동
    fr = torch.randint(0, 16, (N, 2), generator=g).float(); qp = torch.full((N,), 32.)
    cm0 = fake_codemap(N, g); imv = torch.randint(0, 4, (N,), generator=g); merge = torch.randint(0, 2, (N,), generator=g)
    dp = torch.randint(-16, 17, (N, 2), generator=g); bsz = torch.randint(1, 33, (N, 2), generator=g)
    bi = torch.arange(N) % 2 == 0; stt = torch.where(bi, torch.where(torch.arange(N) % 4 == 0, 1, 2), 0)
    with torch.no_grad():
        p = m(Wt, fr, qp, cm0, imv, merge, dp[:, 0], bsz, stt, return_parts=True)
    ref = shift_crop(Wt, fr[:, 0] / 16, fr[:, 1] / 16, 8, 4)
    print('T4 초기 G 최소', float(p['G'].min()), '| Delta=0 가중 평균', float(p['alpha'][:, len(m.dx) // 2].mean()),
          '| 출력 대 단순 이동 최대 차', float((p['y'] - ref).abs().max()) * 1023, '(10비트 단위) | 지도 특징 형상', tuple(p['cf'].shape))
    # T5 양방향 초기 = 두 가설 평균
    net = QKVMC()
    W2 = torch.stack([Wt, Wt.flip(1)], 1); cm2 = torch.stack([cm0, fake_codemap(N, g)], 1)
    b = dict(W=W2, fr=torch.cat([fr, torch.randint(0, 16, (N, 2), generator=g).float()], 1), qp=qp, bi=bi,
             cm=cm2, imv=imv, merge=merge, dpoc=dp, bsz=bsz, st=stt)
    with torch.no_grad():
        y = net(b)
        a0 = net.h(W2[:, 0], b['fr'][:, :2], qp, cm2[:, 0], imv, merge, dp[:, 0], bsz, stt); a1 = net.h(W2[:, 1], b['fr'][:, 2:], qp, cm2[:, 1], imv, merge, dp[:, 1], bsz, stt)
    ib = b['bi']
    print('T5 양방향 초기 대 평균 최대 차:', float((y[ib] - 0.5 * (a0[ib] + a1[ib])).abs().max()) * 1023, '| 단방향 일치:', float((y[~ib] - a0[~ib]).abs().max()) * 1023)
    # T8 L4b: 초기 항등, 커널을 저역통과로 바꾸면 imv=3 & 분수 8 인 축에만 작용
    fr8 = torch.tensor([[8., 8.]]).repeat(N, 1); i3 = torch.full((N,), 3); i0 = torch.zeros(N, dtype=torch.long)
    with torch.no_grad():
        y3 = m(Wt, fr8, qp, cm0, i3, merge, dp[:, 0], bsz, stt); y0 = m(Wt, fr8, qp, cm0, i0, merge, dp[:, 0], bsz, stt)
        d_init = float((y3 - y0).abs().max()) * 1023
        m.alt_k.copy_(torch.tensor([0., 3., 9., 20.]))     # 임의 저역통과 (대체 필터 형태를 빌린 시험값)
        y3b = m(Wt, fr8, qp, cm0, i3, merge, dp[:, 0], bsz, stt); y0b = m(Wt, fr8, qp, cm0, i0, merge, dp[:, 0], bsz, stt)
        frx = torch.tensor([[8., 4.]]).repeat(N, 1); y3x = m(Wt, frx, qp, cm0, i3, merge, dp[:, 0], bsz, stt); y0x = m(Wt, frx, qp, cm0, i0, merge, dp[:, 0], bsz, stt)
        m.alt_k.copy_(torch.tensor([0., 0., 0., 1.]))
    print('T8 L4b 초기 imv3 대 imv0 차', d_init, '| 저역통과 커널 시 차 (양축 8)', float((y3b - y0b).abs().max()) * 1023, '| (x만 8) 차', float((y3x - y0x).abs().max()) * 1023, '| imv0 출력 불변', float((y0b - y0).abs().max()) * 1023)
    # T9 L9 층별 헤드 분리: BDOF 적용 헤드를 흔들어도 BDOF 미적용 블록 출력은 불변
    with torch.no_grad():
        y_a = net(b); net.comb[1][-1].bias.fill_(1.0); net.bcorr[1][-1].bias.fill_(0.01); y_b = net(b)
        net.comb[1][-1].bias.zero_(); net.bcorr[1][-1].bias.zero_()
    s1 = (stt == 1); s2 = (stt == 2)
    print('T9 헤드1 섭동 시 st=1 출력 변화', float((y_a[s1] - y_b[s1]).abs().max()) * 1023, '| st=2 출력 변화', float((y_a[s2] - y_b[s2]).abs().max()) * 1023)
    # T10 위상 차 제약: 두 가설 위상이 같으면 헤드를 흔들어도 평균 고정, 다르면 이탈
    netp = QKVMC(pd_gate=True); netp.load_state_dict(net.state_dict())
    with torch.no_grad():
        netp.comb[0][-1].bias.fill_(1.0); netp.comb[1][-1].bias.fill_(1.0); netp.bcorr[0][-1].bias.fill_(0.01); netp.bcorr[1][-1].bias.fill_(0.01)
        bsame = dict(b); bsame['fr'] = torch.cat([fr, fr], 1); ys = netp(bsame)
        a0s = netp.h(W2[:, 0], fr, qp, cm2[:, 0], imv, merge, dp[:, 0], bsz, stt); a1s = netp.h(W2[:, 1], fr, qp, cm2[:, 1], imv, merge, dp[:, 1], bsz, stt)
        yd = netp(b); a0d = netp.h(W2[:, 0], b['fr'][:, :2], qp, cm2[:, 0], imv, merge, dp[:, 0], bsz, stt); a1d = netp.h(W2[:, 1], b['fr'][:, 2:], qp, cm2[:, 1], imv, merge, dp[:, 1], bsz, stt)
    print('T10 위상 같음: 출력 대 평균 최대 차', float((ys[bi] - 0.5 * (a0s[bi] + a1s[bi])).abs().max()) * 1023, '| 위상 다름: 차', float((yd[bi] - 0.5 * (a0d[bi] + a1d[bi])).abs().max()) * 1023)
    # T11 변형 플래그: FiltAct 제거 시 FiltAct 모듈 없음, no_l9 는 양방향 = 가설 평균, nd=1 step=4 는 후보 9개 범위 ±4/16
    nf = QKVMC(filt=False); nfa = sum(1 for _ in nf.modules() if _.__class__.__name__ == 'FiltAct')
    n9 = QKVMC(nd=1, step=4.0); nl9 = QKVMC(no_l9=True)
    with torch.no_grad():
        yq = nl9(b); c0 = nl9.h(W2[:, 0], b['fr'][:, :2], qp, cm2[:, 0], imv, merge, dp[:, 0], bsz, stt); c1 = nl9.h(W2[:, 1], b['fr'][:, 2:], qp, cm2[:, 1], imv, merge, dp[:, 1], bsz, stt)
    print('T11 filt=False FiltAct 수', nfa, '| nd=1 step=4 후보', len(n9.h.dx), '범위', float(n9.h.dx.min()), float(n9.h.dx.max()), '| no_l9 양방향 대 평균 차', float((yq[bi] - 0.5 * (c0[bi] + c1[bi])).abs().max()) * 1023,
          '| 파라미터 filt=False', sum(p.numel() for p in nf.parameters()), 'ch16', sum(p.numel() for p in QKVMC(ch=16).parameters()), 'nd1', sum(p.numel() for p in n9.parameters()), 'no_l9', sum(p.numel() for p in nl9.parameters()))
    # T12 타일 일반화: 4x4 (20x20 창) 결과가 참조(이전 구현과 동일 경로)와 같은지와 16x16 타일(32x32 창) 전진이 도는지
    with torch.no_grad():
        y4 = m(Wt, fr, qp, cm0, imv, merge, dp[:, 0], bsz, stt)
        W32 = torch.rand(N, 32, 32, generator=g); cm32 = fake_codemap(N, g).repeat(1, 2, 2)[:, :32, :32]
        y16 = m(W32, fr, qp, cm32, imv, merge, dp[:, 0], bsz, stt, tw=16, th=16)
        W20x36 = torch.rand(N, 20, 36, generator=g); cmr = fake_codemap(N, g).repeat(1, 1, 2)[:, :20, :36]
        yr = m(W20x36, fr, qp, cmr, imv, merge, dp[:, 0], bsz, stt, tw=20, th=4)
    print('T12 4x4 출력 형상', tuple(y4.shape), '| 16x16 타일 출력 형상', tuple(y16.shape), '| 20x4 타일 출력 형상', tuple(yr.shape))
    # T6 역전파와 파라미터 수
    T = torch.rand(N, 4, 4, generator=g)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad(); loss = ((net(b) - T) ** 2).mean(); loss.backward(); opt.step()
    ng = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in net.parameters()); nt = len(list(net.parameters()))
    print('T6 (Adam 2단계 후) 파라미터', sum(p.numel() for p in net.parameters()), '| 기울기 0 아닌 텐서', ng, '/', nt)
