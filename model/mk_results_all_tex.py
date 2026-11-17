"""yvvyee: 2초·5-QP 결과를 클래스 여러 개가 든 한 표(EN/KO)로 만든다. 클래스별 표 mk_results_tex.py 의 통합판.
사용: python mk_results_all_tex.py --out tables/results_2s.tex --ko ko/tables/results_2s.tex SPEC [SPEC ...]
SPEC: CLS=raw.txt[,dect.csv[,enct.csv]]   수집된 클래스 (없는 csv 는 생략, 해당 열은 \\planned)
      CLS=planned:Seq1,Seq2,...           결과가 없는 클래스 (전 칸 \\planned)
EncT·DecT 는 다섯 QP 합의 비, 클래스 행은 시퀀스별 값의 평균. \\planned 는 원고 매크로(빨간 planned/예정).
"""
import sys, os, argparse, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdrate import bd_rate
ap = argparse.ArgumentParser(); ap.add_argument('specs', nargs='+'); ap.add_argument('--out'); ap.add_argument('--ko')
ap.add_argument('--splice-en'); ap.add_argument('--splice-ko')  # single-file manuscripts: replace the marked results_2s block
a = ap.parse_args()
sets = {'All': (22, 27, 32, 37, 42), 'Low': (22, 27, 32, 37), 'High': (27, 32, 37, 42)}
cols = [('y', 'All'), ('y', 'Low'), ('y', 'High'), ('u', 'All'), ('v', 'All')]
def fmt(x): x = round(x, 2) + 0.0; return ('$-$%.2f' % -x) if x < 0 else ('%.2f' % x)
def fmtr(x): return r'\planned' if x != x else ('%.0f' % x)
def ratios(path):
    if not path: return {}
    R = collections.defaultdict(lambda: [0.0, 0.0])
    for ln in open(path):
        p = ln.strip().split(',')
        if len(p) >= 5: R[(p[0], p[2])][0] += float(p[3]); R[(p[0], p[2])][1] += float(p[4])
    return {k: v[1] / v[0] for k, v in R.items()}
blocks = []  # (cls, [(name, cells or None)], classcells or None)
ALL = {c: [] for c in ('RA', 'LDB')}; anyplanned = False  # yvvyee: 전체 평균은 모든 시퀀스의 평균, 예정 클래스가 하나라도 있으면 \planned
for spec in a.specs:
    cls, val = spec.split('=', 1)
    if val.startswith('planned:'):
        anyplanned = True
        blocks.append((cls, [(n, None) for n in val[8:].split(',')], None)); continue
    f = val.split(','); DT = ratios(f[1] if len(f) > 1 else None); ET = ratios(f[2] if len(f) > 2 else None)
    D = collections.defaultdict(dict)
    for ln in open(f[0]):
        p = ln.split()
        if len(p) < 10: continue
        seq, qp = p[2].rsplit('_q', 1); D[(p[1], seq)][(p[0], int(qp))] = dict(kbps=float(p[5]), y=float(p[6]), u=float(p[7]), v=float(p[8]))
    seqs = sorted(s for s in {s for (c, s) in D} if s.startswith(cls + '_')); rows = []; acc = {c: [] for c in ('RA', 'LDB')}
    for s in seqs:
        cells = [];
        for c in ('RA', 'LDB'):
            d = D[(c, s)]; v = []
            for k, nm in cols:
                qs = sets[nm]; v.append(bd_rate([d[('anchor', q)]['kbps'] for q in qs], [d[('anchor', q)][k] for q in qs], [d[('model', q)]['kbps'] for q in qs], [d[('model', q)][k] for q in qs]))
            v += [ET.get((c, s), float('nan')), DT.get((c, s), float('nan'))]; acc[c].append(v); ALL[c].append(v)
            cells += [fmt(x) for x in v[:5]] + [fmtr(v[5]), fmtr(v[6])]
        rows.append((s.split('_')[1], cells))
    cc = []
    for c in ('RA', 'LDB'):
        m = np.mean(np.array(acc[c]), axis=0); cc += [fmt(x) for x in m[:5]] + [fmtr(m[5]), fmtr(m[6])]
        print(cls, c, ' '.join('%+.3f' % x for x in m[:5]), 'EncT %.1f DecT %.1f' % (m[5], m[6]))
    blocks.append((cls, rows, cc))
def table(lang):
    cap = (r'BD-rate (\%) of WARP-MC against the VTM-23.8 anchor on the first 2 seconds of the CTC sequences at QP 22, 27, 32, 37, and 42. Y is reported over All-QP, Low-QP (22 to 37), and High-QP (27 to 42), and U and V over All-QP. EncT and DecT are the ratios of the single-thread user times summed over the five QPs. Each class row is the average over the sequences of the class, and the overall row is the average over all sequences.'
           if lang == 'en' else
           r'CTC 시퀀스의 처음 2초를 QP 22, 27, 32, 37, 42로 부호화하여 VTM-23.8 앵커 대비 WARP-MC의 BD-rate(\%)를 구한 결과. Y는 All-QP, Low-QP(22부터 37), High-QP(27부터 42)로, U와 V는 All-QP로 보고한다. EncT와 DecT는 다섯 QP에 대해 합한 단일 스레드 사용자 시간의 비이다. 클래스 행은 해당 클래스 시퀀스의 평균이고, 전체 행은 모든 시퀀스의 평균이다.')
    ra, ldb, seqh, clsw = ('Random Access', 'Low Delay B', 'Sequence', 'Class %s') if lang == 'en' else ('임의 접근', '저지연 B', '시퀀스', '클래스 %s')
    L = [r'\begin{table*}[!t]', r'\caption{%s}' % cap, r'\label{tab:results2s}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l rrrrr rr rrrrr rr@{}}', r'\hline',
         r' & \multicolumn{7}{c}{%s} & \multicolumn{7}{c}{%s} \\' % (ra, ldb),
         r' & \multicolumn{3}{c}{Y} & U & V & \multicolumn{2}{c}{Time} & \multicolumn{3}{c}{Y} & U & V & \multicolumn{2}{c}{Time} \\',
         r'%s & All & Low & High & All & All & EncT & DecT & All & Low & High & All & All & EncT & DecT \\' % seqh, r'\hline']
    pl = r'\multicolumn{14}{c}{\planned}'
    for cls, rows, cc in blocks:
        for n, cells in rows: L.append(n + ' & ' + (' & '.join(cells) if cells else pl) + r' \\')
        L.append(r'\hline'); L.append((clsw % cls) + ' & ' + (' & '.join(cc) if cc else pl) + r' \\'); L.append(r'\hline')
    if anyplanned: oc = None
    else:
        oc = []
        for c in ('RA', 'LDB'):
            m = np.mean(np.array(ALL[c]), axis=0); oc += [fmt(x) for x in m[:5]] + [fmtr(m[5]), fmtr(m[6])]
    L.append(('Overall' if lang == 'en' else '전체') + ' & ' + (' & '.join(oc) if oc else pl) + r' \\'); L.append(r'\hline')
    L += [r'\end{tabular*}', r'\end{table*}']
    return '\n'.join(L) + '\n'
if a.out: open(a.out, 'w').write(table('en'))
if a.ko: open(a.ko, 'w').write(table('ko'))


def splice(path, body):
    b, e = '% >>> begin tables/results_2s', '% <<< end tables/results_2s'
    t = open(path, encoding='utf-8').read()
    i, j = t.index(b), t.index(e)
    open(path, 'w', encoding='utf-8').write(t[:i] + b + '\n' + body.rstrip('\n') + '\n' + t[j:])


if a.splice_en: splice(a.splice_en, table('en'))
if a.splice_ko: splice(a.splice_ko, table('ko'))
