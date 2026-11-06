"""yvvyee: collect_c2s.sh 출력(mode cfg name frames a kbps Y U V YUV time)에서 All/Low/High-QP BD-rate 와 QP별 차를 출력한다. 사용: python bd_c2s.py results.txt"""
import sys, collections, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from bdrate import bd_rate
D=collections.defaultdict(dict)
for ln in open(sys.argv[1]):
    p=ln.split()
    if len(p)<10: continue
    m,c,n=p[0],p[1],p[2]; seq,qp=n.rsplit('_q',1); qp=int(qp)
    D[(c,seq)][(m,qp)]=dict(kbps=float(p[5]),y=float(p[6]),u=float(p[7]),v=float(p[8]),t=float(p[9]))
sets={'All':(22,27,32,37,42),'Low':(22,27,32,37),'High':(27,32,37,42)}
for c in ('RA','LDB'):
    for name,qs in sets.items():
        rows=[]
        for (cc,seq),d in sorted(D.items()):
            if cc!=c: continue
            ra=[d[('anchor',q)]['kbps'] for q in qs]; rb=[d[('model',q)]['kbps'] for q in qs]
            r={k:bd_rate(ra,[d[('anchor',q)][k] for q in qs],rb,[d[('model',q)][k] for q in qs]) for k in 'yuv'}
            et=sum(d[('model',q)]['t'] for q in qs)/sum(d[('anchor',q)]['t'] for q in qs)
            rows.append((seq,r['y'],r['u'],r['v'],et))
        a=np.array([x[1:] for x in rows])
        print(f"{c} {name}: " + "  ".join(f"{s.split('_')[1]} Y{y:+.2f}/U{u:+.2f}/V{v:+.2f}" for s,y,u,v,e in rows) + f"  | mean Y {a[:,0].mean():+.2f} U {a[:,1].mean():+.2f} V {a[:,2].mean():+.2f} | time ratio {a[:,3].mean():.0f}x")
for c in ('RA','LDB'):
    for q in (22,27,32,37,42):
        db=[];dp=[]
        for (cc,seq),d in D.items():
            if cc!=c: continue
            db.append((d[('model',q)]['kbps']/d[('anchor',q)]['kbps']-1)*100); dp.append(d[('model',q)]['y']-d[('anchor',q)]['y'])
        print(f"{c} QP{q}: dRate {np.mean(db):+.3f}%  dYPSNR {np.mean(dp):+.4f} dB")
