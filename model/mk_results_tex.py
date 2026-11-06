"""yvvyee: 2초·5-QP 결과(collect_c2s.sh 출력)에서 원고 표(EN/KO)와 CSV 를 만든다.
사용: python mk_results_tex.py results.txt [--dect dect.csv] [--enct enct.csv] [--out tables/classD_2s.tex --ko ko/tables/classD_2s.tex --csv results_classD_2s.csv]
dect/enct csv: cfg,qp,name,anchor_s,model_s  (EncT·DecT 는 다섯 QP 합의 비, 클래스 행은 시퀀스별 비의 평균)
"""
import sys, argparse, collections, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from bdrate import bd_rate
ap = argparse.ArgumentParser(); ap.add_argument('txt'); ap.add_argument('--dect'); ap.add_argument('--enct')
ap.add_argument('--out'); ap.add_argument('--ko'); ap.add_argument('--csv'); ap.add_argument('--cls', default='D')
a = ap.parse_args()
D = collections.defaultdict(dict)
for ln in open(a.txt):
    p = ln.split()
    if len(p) < 10: continue
    m, c, n = p[0], p[1], p[2]; seq, qp = n.rsplit('_q', 1)
    D[(c, seq)][(m, int(qp))] = dict(kbps=float(p[5]), y=float(p[6]), u=float(p[7]), v=float(p[8]), t=float(p[9]))
def ratios(path):
    R = collections.defaultdict(lambda: [0.0, 0.0])
    if not path: return None
    for ln in open(path):
        p = ln.strip().split(',')
        if len(p) < 5: continue
        R[(p[0], p[2])][0] += float(p[3]); R[(p[0], p[2])][1] += float(p[4])
    return {k: v[1] / v[0] for k, v in R.items()}
DT, ET = ratios(a.dect), ratios(a.enct)
sets = {'All': (22, 27, 32, 37, 42), 'Low': (22, 27, 32, 37), 'High': (27, 32, 37, 42)}
seqs = sorted({s for (c, s) in D if s.startswith(a.cls + '_')}, key=lambda s: ['BQSquare','BasketballPass','BlowingBubbles','RaceHorses'].index(s.split('_')[1]) if s.split('_')[1] in ['BQSquare','BasketballPass','BlowingBubbles','RaceHorses'] else 99)
rows = {}  # (cfg, seq) -> dict
for c in ('RA', 'LDB'):
    for s in seqs:
        d = D[(c, s)]
        if any((m, q) not in d for m in ('anchor', 'model') for q in sets['All']): print('incomplete', c, s); continue
        r = {}
        for nm, qs in sets.items():
            ra = [d[('anchor', q)]['kbps'] for q in qs]; rb = [d[('model', q)]['kbps'] for q in qs]
            for k in 'yuv': r[(k, nm)] = bd_rate(ra, [d[('anchor', q)][k] for q in qs], rb, [d[('model', q)][k] for q in qs])
        r['enct'] = ET.get((c, s), float('nan')) if ET else float('nan')
        r['dect'] = DT.get((c, s), float('nan')) if DT else float('nan')
        rows[(c, s)] = r
cols = [('y','All'),('y','Low'),('y','High'),('u','All'),('v','All')]
def fmt(x): return ('$-$%.2f' % -x) if x < 0 else ('%.2f' % x)
def fmtr(x): return '--' if x != x else ('%.0f' % x)
def mean(c, key): v = [rows[(c, s)][key] for s in seqs if (c, s) in rows]; return float(np.mean(v)) if v else float('nan')
def table(lang):
    cap_en = r'BD-rate (\%) of WARP-MC against the VTM-23.8 anchor on the first 2 seconds of the class D sequences at QP 22, 27, 32, 37, and 42. Y is reported over All-QP, Low-QP (22 to 37), and High-QP (27 to 42), and U and V over All-QP. EncT and DecT are the ratios of the single-thread user times summed over the five QPs.'
    cap_ko = r'클래스 D 시퀀스의 처음 2초를 QP 22, 27, 32, 37, 42로 부호화하여 VTM-23.8 앵커 대비 WARP-MC의 BD-rate(\%)를 구한 결과. Y는 All-QP, Low-QP(22부터 37), High-QP(27부터 42)로, U와 V는 All-QP로 보고한다. EncT와 DecT는 다섯 QP에 대해 합한 단일 스레드 사용자 시간의 비이다.'
    ra, ldb, seqh, clsh = ('Random Access', 'Low Delay B', 'Sequence', 'Class ' + a.cls) if lang == 'en' else ('임의 접근', '저지연 B', '시퀀스', '클래스 ' + a.cls)
    L = [r'\begin{table*}[!t]', r'\caption{%s}' % (cap_en if lang == 'en' else cap_ko), r'\label{tab:classD2s}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}',
         r'\begin{tabular}{l rrrrr rr rrrrr rr}', r'\hline',
         r' & \multicolumn{7}{c}{%s} & \multicolumn{7}{c}{%s} \\' % (ra, ldb),
         r' & \multicolumn{3}{c}{Y} & U & V & \multicolumn{2}{c}{Time} & \multicolumn{3}{c}{Y} & U & V & \multicolumn{2}{c}{Time} \\',
         r'%s & All & Low & High & All & All & EncT & DecT & All & Low & High & All & All & EncT & DecT \\' % seqh, r'\hline']
    for s in seqs:
        cells = [s.split('_')[1]]
        for c in ('RA', 'LDB'):
            r = rows[(c, s)]; cells += [fmt(r[k]) for k in cols] + [fmtr(r['enct']), fmtr(r['dect'])]
        L.append(' & '.join(cells) + r' \\')
    L.append(r'\hline'); cells = [clsh]
    for c in ('RA', 'LDB'):
        cells += [fmt(mean(c, k)) for k in cols] + [fmtr(mean(c, 'enct')), fmtr(mean(c, 'dect'))]
    L += [' & '.join(cells) + r' \\', r'\hline', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(L) + '\n'
if a.out: open(a.out, 'w').write(table('en'))
if a.ko: open(a.ko, 'w').write(table('ko'))
if a.csv:
    with open(a.csv, 'w') as f:
        f.write('cfg,seq,Y_All,Y_Low,Y_High,U_All,U_Low,U_High,V_All,V_Low,V_High,EncT,DecT\n')
        for (c, s), r in rows.items():
            f.write(','.join([c, s] + ['%.4f' % r[(k, nm)] for k in 'yuv' for nm in sets] + ['%.3f' % r['enct'], '%.3f' % r['dect']]) + '\n')
print(table('en'))
for c in ('RA', 'LDB'): print(c, 'mean', ' '.join('%s_%s=%+.3f' % (k, nm, mean(c, (k, nm))) for k, nm in cols), 'EncT=%.1f DecT=%.1f' % (mean(c, 'enct'), mean(c, 'dect')))
