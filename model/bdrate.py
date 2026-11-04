"""yvvyee: VTM 부호화 로그에서 비트율·PSNR 을 읽어 Bjontegaard BD-rate(조각별 3차, 4 QP) 를 계산한다.
사용: python bdrate.py anchor_dir test_dir [--cfg RA]
  각 디렉터리에 <seq>_q<QP>.log (EncoderApp 로그) 가 있어야 한다. LayerId 0 의 'Total Frames | Bitrate Y-PSNR U-PSNR V-PSNR' 표(VTM-23.8 형식)를 쓴다.
출력: 시퀀스별 Y/U/V BD-rate(%) 와 클래스 평균, 부호화 시간 비.
"""
import sys, os, re, glob, collections
import numpy as np

def parse(f):
    txt = open(f, errors='ignore').read()
    m = re.search(r'Total Frames \|\s*Bitrate\s+Y-PSNR\s+U-PSNR\s+V-PSNR.*\n\s*(\d+)\s+a\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', txt)
    if not m: return None
    t = re.search(r'Total Time:\s+([\d.]+) sec', txt)
    return dict(frames=int(m.group(1)), kbps=float(m.group(2)), y=float(m.group(3)), u=float(m.group(4)), v=float(m.group(5)), time=float(t.group(1)) if t else float('nan'))

def bd_rate(r1, p1, r2, p2):
    """조각별 3차 에르미트 보간(JVET 공통 BD-rate 계산)으로 두 곡선의 비트율 차(%)."""
    lr1, lr2 = np.log(r1), np.log(r2)
    lo, hi = max(min(p1), min(p2)), min(max(p1), max(p2))
    def pchip_int(x, y, a, b):
        from scipy.interpolate import PchipInterpolator
        o = np.argsort(x); f = PchipInterpolator(np.asarray(x)[o], np.asarray(y)[o])
        return f.integrate(a, b)
    avg = (pchip_int(p2, lr2, lo, hi) - pchip_int(p1, lr1, lo, hi)) / (hi - lo)
    return (np.exp(avg) - 1) * 100

def main():
    a, b = sys.argv[1], sys.argv[2]
    seqs = sorted(set(re.sub(r'_q\d+\.log$', '', os.path.basename(f)) for f in glob.glob(f'{b}/*_q*.log')))
    rows = []; per_class = collections.defaultdict(list)
    for s in seqs:
        A = {}; B = {}
        for qp in (22, 27, 32, 37):
            fa, fb = f'{a}/{s}_q{qp}.log', f'{b}/{s}_q{qp}.log'
            if os.path.exists(fa) and os.path.exists(fb):
                pa, pb = parse(fa), parse(fb)
                if pa and pb: A[qp] = pa; B[qp] = pb
        if len(A) < 4: print(f'{s}: incomplete ({len(A)}/4)'); continue
        qs = sorted(A); ra = [A[q]['kbps'] for q in qs]; rb = [B[q]['kbps'] for q in qs]
        res = {c: bd_rate(ra, [A[q][c] for q in qs], rb, [B[q][c] for q in qs]) for c in 'yuv'}
        et = sum(B[q]['time'] for q in qs) / sum(A[q]['time'] for q in qs)
        rows.append((s, res['y'], res['u'], res['v'], et)); per_class[s[0]].append((res['y'], res['u'], res['v'], et))
        print(f'{s:32s} Y {res["y"]:+7.2f}  U {res["u"]:+7.2f}  V {res["v"]:+7.2f}  EncT {et:6.1f}x')
    for c, v in sorted(per_class.items()):
        v = np.array(v); print(f'Class {c} ({len(v)} seq)        Y {v[:,0].mean():+7.2f}  U {v[:,1].mean():+7.2f}  V {v[:,2].mean():+7.2f}  EncT {v[:,3].mean():6.1f}x')

if __name__ == '__main__': main()
