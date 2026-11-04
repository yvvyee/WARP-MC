"""yvvyee: C++ 고정소수 커널용 내보내기 — L9 없는 가설별 모형(Hypothesis, L1~L8)의 양자화 파라미터·고정 FIR·조회표·층별 스케일과,
덤프 레코드 일부에 대한 기준 출력(가설 0, Q10.4 정수)을 디렉터리에 쓴다. 모사기(qkv_fixed.py, float64, 양자화 전부 켬)와 같은 산술이어야 한다.
사용: python qkv_export.py ckpt.pt calib.dmp out_dir [n_ref]
형식: manifest.txt 의 각 줄 = name dtype ndim d0 d1 ... file ; 데이터는 리틀엔디언 원시 파일
"""
import os, sys, math, numpy as np, torch
sys.argv.append('--f64')
import qkv_fixed as X
from qkv_fixed import Q, HypQ, load_dump, batches
from qkv_mc import QKVMC, gauss1d, kaiser_frac, WS

torch.set_num_threads(8); torch.set_grad_enabled(False)
ck, calib, out = sys.argv[1], sys.argv[2], sys.argv[3]; n_ref = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else 512
os.makedirs(out, exist_ok=True)
man = []
def put(name, arr, dtype):
    a = np.ascontiguousarray(np.asarray(arr), dtype=dtype); fn = f'{name}.bin'; a.tofile(os.path.join(out, fn))
    man.append(f'{name} {np.dtype(dtype).name} {a.ndim} ' + ' '.join(str(d) for d in a.shape) + f' {fn}')
def exp2int(sc): e = int(round(math.log2(sc))); assert 2.0 ** e == sc, sc; return e

sd = torch.load(ck, map_location='cpu'); net = QKVMC(no_l9=True, ch=X.ckpt_ch(sd)); net.load_state_dict(sd); net.eval(); net.double(); h = net.h; hq = HypQ(h)
M, W, T, C0, H2, CM, st, bi = load_dump([calib])
# ---- 보정 (층별 활성·에너지 스케일): 첫 4096 레코드
with torch.no_grad():
    Q.calib = True; n = 0
    for j, b in batches(M, W, H2, CM, st, bi):
        fr = b['fr'].clamp(min=0)
        hq.forward(b['W'][:, 0], fr[:, 0:2], b['qp'], b['cm'][:, 0], b['imv'], b['merge'], b['dpoc'][:, 0], b['bsz'], b['st']); n += j.stop - j.start
        if n >= 4096: break
    Q.calib = False
    # 스케일 확정 (한 번 통과)
    j, b = next(iter(batches(M, W, H2, CM, st, bi, bs=64))); fr = b['fr'].clamp(min=0)
    hq.forward(b['W'][:, 0], fr[:, 0:2], b['qp'], b['cm'][:, 0], b['imv'], b['merge'], b['dpoc'][:, 0], b['bsz'], b['st'])

# ---- 스칼라·상수
put('consts_i32', [WS, h.nb, len(h.dx), 12, Q.pix_frac, Q.fir_frac, int(h.rate_gate)], np.int32)
put('cand_dx_q4', (h.dx * 16).round().numpy(), np.int32); put('cand_dy_q4', (h.dy * 16).round().numpy(), np.int32)   # 후보 오프셋 (1/16 화소 단위)
put('const_log2e_q16', [int(round(Q.LOG2E_Q * 2 ** 16))], np.int32); put('const_ln2_q16', [int(round(Q.LN2_Q * 2 ** 16))], np.int32)
put('const_inv30_q16', [int(round(2 ** 16 / 30.0))], np.int32)
# ---- 고정 FIR (Q1.14)
for nm, hh in (('gauss06', gauss1d(0.6, 2)), ('gauss12', gauss1d(1.2, 4)), ('gauss24', gauss1d(2.4, 8)), ('gauss10', gauss1d(1.0, 2))):
    put('fir_' + nm, (Q.fir(hh.double()) * 2 ** Q.fir_frac).round().numpy(), np.int16)
fa = [m for m in h.corr if m.__class__.__name__ == 'FiltAct'][0]
put('fir_filtact_hu_half', (Q.fir(fa.hu.double() / 2) * 2 ** Q.fir_frac).round().numpy(), np.int16); put('fir_filtact_hd', (Q.fir(fa.hd.double()) * 2 ** Q.fir_frac).round().numpy(), np.int16)
ph = torch.arange(16, dtype=torch.float64) / 16
put('fir_kaiser16x12', (X.kaiser_frac_q(ph, 12) * 2 ** Q.fir_frac).round().numpy(), np.int16)
k = torch.cat([h.alt_k, h.alt_k[:3].flip(0)]).double(); put('alt_kernel7', (Q.fir(k / k.sum()) * 2 ** Q.fir_frac).round().numpy(), np.int16)
put('alt_gate_q15', [int(round(float(Q.rnd(torch.sigmoid(h.alt_g.double()), 15, bits=32)) * 2 ** 15))], np.int32)
# ---- 율 문턱: QP 별 th (화소 단위, Q10.4) = Q.pix(exp(tau) * 4*2^((qp-4)/6)/1023) * 1023 * 16
qps = torch.arange(64, dtype=torch.float64); th = Q.pix(torch.exp(h.tau.double()) * (4 * 2 ** ((qps - 4) / 6) / X.PIXN))
put('rate_th_q4_by_qp', (th * X.PIXN * 16).round().numpy(), np.int32)
# ---- 학습 행렬: phi (K x 5), D4 (4x4): int16 + 지수
for nm, t in (('phi', h.phi), ('d4', h.D4)):
    w = t.double(); m = w.abs().max(); e = int(torch.floor(torch.log2(32767.0 / m)).item()); put(f'mat_{nm}_i16', (Q.weight(w) * 2 ** e).round().numpy(), np.int16); put(f'mat_{nm}_exp', [e], np.int32)
# ---- 합성곱 층: weight int16 + exp, bias Q1.14 int32, 출력 스케일 지수(보정), FiltAct 출력 스케일
import torch.nn as nn
layers = [('noise', h.noise), ('dq', h.dq), ('corr', h.corr)]
conv_list = []
for gname, seq in layers:
    for i, m in enumerate(seq):
        if isinstance(m, nn.Conv2d):
            w = m.weight.double(); mx = w.abs().max(); e = int(torch.floor(torch.log2(32767.0 / mx)).item())
            put(f'conv_{gname}_{i}_w', (Q.weight(w) * 2 ** e).round().numpy(), np.int16); put(f'conv_{gname}_{i}_wexp', [e], np.int32)
            put(f'conv_{gname}_{i}_b_q14', (Q.act(m.bias.double()) * 2 ** 14).round().numpy(), np.int32)
            put(f'conv_{gname}_{i}_oexp', [exp2int(Q.scale[id(m)])], np.int32)
            conv_list.append(f'{gname}_{i} k{m.kernel_size[0]} in{m.in_channels} out{m.out_channels} pad{m.padding[0]} {m.padding_mode}')
        elif m.__class__.__name__ == 'FiltAct':
            put(f'filtact_{gname}_{i}_oexp', [exp2int(Q.scale[(id(m), 'h')]), exp2int(Q.scale[(id(m), 'v')])], np.int32)
open(os.path.join(out, 'layers.txt'), 'w').write('\n'.join(conv_list) + '\n')
put('corr_in_exp', [exp2int(Q.scale['corr_in'])], np.int32)
# log2(bsz)/5 표 (bsz 단위 1..32, Q1.14) 와 qp/63, |dpoc|/16, (qpr-qp)/12 는 정수식으로 재현
bs = torch.arange(1, 33, dtype=torch.float64); put('tab_log2bsz_q14', (Q.act(torch.log2(bs) / 5.0) * 2 ** 14).round().numpy(), np.int32)
# ---- 에너지 스케일 지수
for key in (('sq', 0), ('sq', 1), ('sq', 2), ('sq', 3), ('phi', 0), ('phi', 1), ('phi', 2), ('phi', 3), 'st0', 'st', 'phin'):
    nm = key if isinstance(key, str) else f'{key[0]}{key[1]}'
    put(f'energy_{nm}_exp', [exp2int(Q.scale[('e', key)])], np.int32)
# ---- 조회표
f8 = torch.arange(256, dtype=torch.float64) / 256; put('lut_exp2_frac_q15', (Q.rnd(torch.exp2(f8), 15, bits=32) * 2 ** 15).round().numpy(), np.int32)
m10 = 1 + torch.arange(1025, dtype=torch.float64) / 1024          # 가수 Q10 반올림이 2.0 이 될 수 있어 1025 항목
put('lut_log2_mant_q10', (Q.rnd(torch.log2(m10), 10, bits=32) * 2 ** 10).round().numpy(), np.int32)
put('lut_recip_mant_q15', (Q.rnd(1.0 / m10, 15, bits=32) * 2 ** 15).round().numpy(), np.int32)
x8 = torch.arange(-16 * 256, 16 * 256 + 1, dtype=torch.float64) / 256; put('lut_sigmoid_q15', (Q.rnd(torch.sigmoid(x8), 15, bits=32) * 2 ** 15).round().numpy(), np.int32)
# ---- 기준 출력: 처음 n_ref 레코드의 가설 0 (y 4x4, y_lin 8x8, xd 20x20; Q10.4 정수)
ys, yl, xds, idx = [], [], [], []
with torch.no_grad():
    for j, b in batches(M, W, H2, CM, st, bi, bs=128):
        fr = b['fr'].clamp(min=0)
        p = hq.forward(b['W'][:, 0], fr[:, 0:2], b['qp'], b['cm'][:, 0], b['imv'], b['merge'], b['dpoc'][:, 0], b['bsz'], b['st'])
        ys.append((p['y'] * X.PIXN * 16).round().numpy()); yl.append((p['y_lin'] * X.PIXN * 16).round().numpy()); xds.append((p['xd'] * X.PIXN * 16).round().numpy()); idx.append(np.arange(j.start, j.stop))
        if j.stop >= n_ref: break
put('ref_idx', np.concatenate(idx)[:n_ref], np.int32); put('ref_y_q4', np.concatenate(ys)[:n_ref], np.int32); put('ref_ylin_q4', np.concatenate(yl)[:n_ref], np.int32); put('ref_xd_q4', np.concatenate(xds)[:n_ref], np.int32)
open(os.path.join(out, 'manifest.txt'), 'w').write('\n'.join(man) + '\n')
print('exported', len(man), 'arrays to', out, '| conv layers:', len(conv_list), '| ref records', min(n_ref, len(np.concatenate(idx))))
