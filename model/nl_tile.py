"""yvvyee: 타일 의미(stage4_integration.md 11절) 학습·평가 — L9 없는 가설별 QKV-MC 를 타일 레코드(4640)로 학습 (server19, 측정 전용).

레코드: 머리부 32 [st, qp, poc, px, py, pw, ph, tx, ty, tw, th, fx0, fy0, fx1, fy1, go0, go1, imv, merge, dpoc0, dpoc1, nh],
        가설별 32x32 정수 창 2, 가설별 32x32 지도 창 2, 16x16 원본, 16x16 VTM 예측. 창은 (th+16)x(tw+16) 을 stride 32 로 둔다.
학습: bvi_orig/dump_tile (콘텐츠 CRC 10 % 검증), 시험: ctc23/dump_tile. 타일 크기 (tw,th) 별로 배치를 묶는다.
C0: 6.1절 재현기로 계산 (단방향은 isLast 경로, 양방향은 14비트 평균). 지표: C0 대비 SSE %, 층별·타일 크기별.
환경변수: SEED, EPOCHS(25), PATIENCE(6), BS(128), KEEP_FRAC(1), TRAIN_PIX(1e8 출력 화소), STRATA_W(sqrt), ND, STEP, FILT, CH, DEPTH, TAG_SUFFIX, OUTDIR
"""
import glob, os, json, time, zlib, re, math, sys
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qkv_mc import QKVMC
import bdof_replica as R

torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
BASE = '/home/ywlee/dev/oracle_runs'; REC = 4640; TW = 32; TT = 16
dev = torch.device('cuda')
SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); np.random.seed(SEED)
CH = int(os.environ.get('CH', '32')); DEPTH = int(os.environ.get('DEPTH', '5')); ND = int(os.environ.get('ND', '2')); STEP = float(os.environ.get('STEP', '2.0')); FILT = os.environ.get('FILT', '1') == '1'
EPOCHS = int(os.environ.get('EPOCHS', '25')); PATIENCE = int(os.environ.get('PATIENCE', '6')); BS = int(os.environ.get('BS', '128'))
KEEP_FRAC = float(os.environ.get('KEEP_FRAC', '1')); TRAIN_PIX = float(os.environ.get('TRAIN_PIX', '1e8')); STRATA_W = os.environ.get('STRATA_W', 'sqrt')
LIMIT = int(os.environ.get('LIMIT_FILES', '0'))
TAG = f'TILE_s{SEED}_c{CH}_d{DEPTH}_nd{ND}' + os.environ.get('TAG_SUFFIX', '')
OUT = os.environ.get('OUTDIR', f'{BASE}/bli/eval5'); os.makedirs(OUT, exist_ok=True)
# yvvyee: 중복 기동 방지 — 같은 TAG 의 결과나 잠금이 있으면 종료 (대기열 둘이 같은 시드를 띄우는 경우 대비)
LOCK = f'{OUT}/lock_{TAG}'
if os.path.exists(f'{OUT}/res_{TAG}.json') or os.path.exists(LOCK): print(f'skip {TAG}: result or lock exists', flush=True); sys.exit(0)
open(LOCK, 'w').write(f'{os.getpid()} {time.ctime()}\n')
import atexit; atexit.register(lambda: os.path.exists(LOCK) and os.remove(LOCK))


def content(f):
    m = re.search(r'_([A-D])([A-Za-z0-9]+?)_(\d+)x(\d+)_', os.path.basename(f)); return m.group(2) if m else os.path.basename(f)


class Data:
    """타일 레코드를 필드별로 보관하고 (tw,th) 그룹 색인을 만든다."""
    def __init__(self, files, keyfn, keep=1.0):
        H, Wn, Mp, T, V, keys = [], [], [], [], [], []
        for f in files:
            a = np.fromfile(f, dtype=np.int16)
            if a.size % REC: continue
            a = a.reshape(-1, REC)
            if keep < 1.0:
                rs = np.random.RandomState(zlib.crc32(os.path.basename(f).encode()) & 0xFFFFFFFF); a = a[rs.rand(len(a)) < keep]
            H.append(a[:, :22].copy()); Wn.append(a[:, 32:32 + 2048].reshape(-1, 2, TW, TW).copy()); Mp.append(a[:, 32 + 2048:32 + 4096].reshape(-1, 2, TW, TW).copy())
            T.append(a[:, 32 + 4096:32 + 4096 + 256].reshape(-1, TT, TT).copy()); V.append(a[:, 32 + 4096 + 256:].reshape(-1, TT, TT).copy()); keys += [keyfn(f)] * len(a)
        self.H = np.concatenate(H); self.W = np.concatenate(Wn); self.M = np.concatenate(Mp); self.T = np.concatenate(T); self.V = np.concatenate(V); self.keys = np.array(keys)
        self.st = self.H[:, 0].astype(np.int64); self.bi = self.st > 0; self.tw = self.H[:, 9].astype(int); self.th = self.H[:, 10].astype(int)
        self.npix = self.tw * self.th
        self.groups = {}
        for i, (w, h) in enumerate(zip(self.tw, self.th)): self.groups.setdefault((int(w), int(h)), []).append(i)
        self.groups = {k: np.array(v) for k, v in self.groups.items()}
        self.C0 = None

    def compute_c0(self):
        """C0 = DCTIF 예측 (BDOF 없음): 단방향은 isLast 화소 경로, 양방향은 14비트 두 가설의 VTM 평균. 재현기로 계산하고 층 0·1 에서는 VTM 예측과 비트 일치해야 한다."""
        self.C0 = np.zeros((len(self.H), TT, TT), np.float32); eq = np.zeros(len(self.H), bool)
        for (w, h), idx in self.groups.items():
            for s in range(0, len(idx), 4096):
                j = idx[s:s + 4096]; fr = np.maximum(self.H[j, 11:15].astype(np.int64), 0); alt = self.H[j, 17] == 3; nh = self.H[j, 21]
                win = self.W[j, :, :h + 16, :w + 16]
                p0 = R.vtm_pred14_rect(win[:, 0], fr[:, 0], fr[:, 1], alt, 8, h, w); p1 = R.vtm_pred14_rect(win[:, 1], fr[:, 2], fr[:, 3], alt, 8, h, w)
                c = np.where((nh > 1)[:, None, None], R.bi_avg(p0, p1), R.uni_from14(p0))
                self.C0[j, :h, :w] = c; eq[j] = (c == self.V[j, :h, :w]).all((1, 2))
        chk = self.st < 2
        print(f'C0 check (st 0/1, must be all equal): {int(eq[chk].sum())}/{int(chk.sum())} | st2 equal (BDOF off, expected < all): {int(eq[~chk].sum())}/{int((~chk).sum())}', flush=True)
        assert eq[chk].all(), 'C0 재현이 VTM 예측과 불일치'

    def batch(self, idx):
        w, h = int(self.tw[idx[0]]), int(self.th[idx[0]]); g = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)
        Hd = self.H[idx].astype(np.int64)
        return dict(W=g(self.W[idx, :, :h + 16, :w + 16].astype(np.float32)) / 1023.0, fr=g(Hd[:, 11:15].astype(np.float32)), qp=g(Hd[:, 1].astype(np.float32)),
                    bi=g(self.bi[idx]), imv=g(Hd[:, 17]), merge=g(Hd[:, 18]), dpoc=g(Hd[:, 19:21]), bsz=g(np.stack([Hd[:, 5] >> 2, Hd[:, 6] >> 2], 1)),
                    cm=g(self.M[idx, :, :h + 16, :w + 16].astype(np.int32)), st=g(self.st[idx]), T=g(self.T[idx, :h, :w].astype(np.float32)), tw=w, th=h)


def complete(f):
    try: return 'records' in open(f + '.log').read()
    except OSError: return False


files = [f for f in sorted(glob.glob(os.environ.get('TRAIN_GLOB', f'{BASE}/bvi_orig/dump_tile/*.dmp'))) if complete(f)]
test_files = [f for f in sorted(glob.glob(os.environ.get('TEST_GLOB', f'{BASE}/ctc23/dump_tile/*.dmp'))) if complete(f)]
if LIMIT: files = files[:LIMIT]; test_files = test_files[:max(2, LIMIT // 10)]
TR = Data(files, content, KEEP_FRAC); TE = Data(test_files, lambda f: os.path.basename(f)[:-4])
isval = np.array([zlib.crc32(k.encode()) % 10 == 0 for k in TR.keys])
if isval.all() or not isval.any(): isval = np.random.RandomState(7).rand(len(isval)) < 0.1; print('degenerate split -> random 10 % val', flush=True)
itr = np.where(~isval)[0]; iva = np.where(isval)[0]
# 학습 표본은 출력 화소 예산으로 제한 (고정 난수)
rs = np.random.RandomState(1234); perm = rs.permutation(itr); cum = np.cumsum(TR.npix[perm]); itr = np.sort(perm[:int(np.searchsorted(cum, TRAIN_PIX)) + 1]) if cum[-1] > TRAIN_PIX else itr
if len(iva) > 100000: iva = np.sort(np.random.RandomState(1235).choice(iva, 100000, replace=False))
cnt = np.array([TR.npix[itr][TR.st[itr] == s].sum() for s in range(3)], np.float64)
SW = torch.ones(3, device=dev)
if STRATA_W == 'sqrt':
    w = 1 / np.sqrt(np.maximum(cnt, 1)); w = w * cnt.sum() / (cnt * w).sum(); SW = torch.tensor(w, dtype=torch.float32, device=dev)
print(f'{TAG} train recs {len(itr)} pix {int(TR.npix[itr].sum())} | val recs {len(iva)} | test recs {len(TE.H)} | strata_pix {cnt.astype(int).tolist()} strata_w {SW.tolist()} | groups {sorted((k, int((TR.tw[itr] == k[0]).sum())) for k in TR.groups)}', flush=True)
TR.compute_c0(); TE.compute_c0()
net = QKVMC(ch=CH, depth=DEPTH, nd=ND, step=STEP, filt=FILT, no_l9=True).to(dev)
print('params', sum(p.numel() for p in net.parameters()), flush=True)
opt = torch.optim.Adam(net.parameters(), lr=float(os.environ.get('LR', '1e-3')))
# 그룹별 배치 목록 (epoch 마다 섞음)
def make_batches(ds, idx, bs, shuffle):
    out = []
    for k, gidx in ds.groups.items():
        sel = np.intersect1d(gidx, idx, assume_unique=False)
        if shuffle: sel = sel[np.random.permutation(len(sel))]
        for i in range(0, len(sel), bs): out.append(sel[i:i + bs])
    if shuffle: out = [out[i] for i in np.random.permutation(len(out))]
    return out
steps = sum(1 for _ in make_batches(TR, itr, BS, False)); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * steps)


def loss_fn(pred, b):
    e = pred - b['T'] / 1023.0; w = SW[b['st'].long()]
    return (w * (e ** 2).mean((1, 2))).mean()


def predict(ds, idx):
    out = np.zeros((len(ds.H), TT, TT), np.float32)
    with torch.no_grad():
        for j in make_batches(ds, idx, 256, False):
            b = ds.batch(j); out[j, :b['th'], :b['tw']] = (net(b) * 1023.0).cpu().numpy()
    return out


def sse(ds, idx, P):
    m = np.zeros((len(ds.H), TT, TT), bool)
    for i in idx: m[i, :ds.th[i], :ds.tw[i]] = True
    T = ds.T.astype(np.float32); return float((((T - P) ** 2) * m)[idx].sum()), float((((T - ds.C0) ** 2) * m)[idx].sum()), float((((T - ds.V.astype(np.float32)) ** 2) * m)[idx].sum())


best, best_ep, best_state, wait, t0 = 1e30, -1, None, 0, time.time()
for ep in range(EPOCHS):
    net.train(); tot = 0; n = 0
    for j in make_batches(TR, itr, BS, True):
        b = TR.batch(j); loss = loss_fn(net(b), b)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); tot += loss.item() * len(j); n += len(j)
    net.eval(); pv = predict(TR, iva); vs, v0, _ = sse(TR, iva, pv)
    print(f'ep{ep} loss {tot / n:.6f} val {100 * (vs / v0 - 1):+.3f}% tau {net.h.tau.item():.3f} {time.time() - t0:.0f}s', flush=True)
    if vs < best: best, best_ep, wait = vs, ep, 0; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE: break
net.load_state_dict(best_state); net.eval()
P = predict(TE, np.arange(len(TE.H))); torch.save(best_state, f'{OUT}/net_{TAG}.pt'); np.save(f'{OUT}/pred_{TAG}.npy', P)
CL = np.array([k[0] for k in TE.keys]); QP = np.array([int(m.group(1)) if (m := re.search(r'_q(\d+)', k)) else -1 for k in TE.keys])
res = {'tag': TAG, 'best_epoch': best_ep, 'params': sum(p.numel() for p in net.parameters()), 'rows': []}
def rel(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0: return None
    s, s0, sv = sse(TE, idx, P); return dict(n=int(len(idx)), pix=int(TE.npix[idx].sum()), nn=100 * (s / s0 - 1), vtm=100 * (sv / s0 - 1))
for s_ in (0, 1, 2):
    for key, vals in (('all', [None]), ('class', sorted(set(CL))), ('qp', [22, 27, 32, 37]), ('tile', sorted(TE.groups))):
        for v in vals:
            m = TE.st == s_
            if key == 'class': m &= CL == v
            if key == 'qp': m &= QP == v
            if key == 'tile': m &= (TE.tw == v[0]) & (TE.th == v[1])
            r = rel(m)
            if r: res['rows'].append(dict(stratum=s_, key=key, val=None if v is None else str(v), **r))
    r = [x for x in res['rows'] if x['stratum'] == s_ and x['key'] == 'all'][0]
    print(f'TEST stratum {s_} pix={r["pix"]} NN {r["nn"]:+.3f}% VTM {r["vtm"]:+.3f}%', flush=True)
json.dump(res, open(f'{OUT}/res_{TAG}.json', 'w'), indent=1)
