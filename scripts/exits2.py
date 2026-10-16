import numpy as np, pandas as pd, yfinance as yf
src=open('scripts/count_sig.py').read().split("tot=sum(")[0]
for a,b in [("aF<=0.75*aS","aF<=0.85*aS"),("<=0.75*aS)&","<=1.0*aS)&"),("<=0.6*aS)","<=0.8*aS)"),("r>=1.3*aF","r>=1.0*aF"),(">=0.7\n",">=0.65\n"),("V.iat[i]>=1.5*vA","V.iat[i]>=1.2*vA"),("pb>=6","pb>=5"),("i-cb<=5 and","i-cb<=10 and"),("elif i-cb>5","elif i-cb>10"),(" and (c-cl)/c<=0.08)",")")]:
    assert a in src, a; src=src.replace(a,b)
src=src.replace("                if i>=start: n+=1","                if i>=start: SIG.append((s,df.index[i],c,cl,i,df,e21))")
SIG=[]; exec(compile(src,'x','exec'))
syms=[s for s in syms if not s.startswith("MON")]
for s in syms: run(s)
n=yf.download('^NSEI',start='2025-06-01',progress=False,auto_adjust=True)['Close'].squeeze()
mkt=(n>n.ewm(span=50,adjust=False).mean())
def sim(MAXSTOP, BE, PT, PF, cost=0.0015):
    rows=[]
    for s,d,entry,cl,i,df,e21 in SIG:
        H,L,C,O,E=df.High.values,df.Low.values,df.Close.values,df.Open.values,e21.values
        stop=max(cl, entry*(1-MAXSTOP)) if MAXSTOP else cl
        R=entry-stop; sp=stop; be=False; part=False; pnl=0.0; rem=1.0; why="open"; ex=C[-1]; best=0
        j=i+1
        while j<len(df):
            if O[j]<=sp: ex=O[j]; why="BE" if be else "stop"; break
            if L[j]<=sp: ex=sp; why="BE" if be else "stop"; break
            if PT and not part and H[j]>=entry+PT*R:
                pnl+=PF*(max(O[j],entry+PT*R)-entry); rem-=PF; part=True
            if BE and not be and H[j]>=entry+BE*R: be=True; sp=max(sp,entry)
            if C[j]<E[j] and j+1<len(df): ex=O[j+1]; why="21EMA"; break
            j+=1
        pnl+=rem*(ex-entry); pnl-=2*cost*entry
        rows.append(dict(sym=s,date=d,stop_pct=R/entry*100,R=pnl/R,ret=pnl/entry*100,why=why,mkt=bool(mkt.asof(d))))
    return pd.DataFrame(rows)
def show(name,t):
    for lab,g in [("all",t),("NIFTY>EMA50",t[t.mkt])]:
        w=g[g.R>0]
        print(f"{name:28s} {lab:12s} n={len(g):3d} win={len(w)/len(g)*100:3.0f}% avgW={w.R.mean():5.2f}R avgL={g[g.R<=0].R.mean():5.2f}R exp={g.R.mean():+.2f}R sum={g.R.sum():+6.1f}R ~cap@0.5%={g.R.sum()*0.5:+5.1f}% bestR={g.R.max():.1f}")
new=sim(0.025,2,4,0.25); show("NEW 2.5%stop BE2R 25%@4R",new)
show("OLD contrLow 33%@3R BE",sim(0,3,3,0.33))
print(); print(new.why.value_counts().to_string())
print(new[new.mkt].sort_values('R',ascending=False).head(6)[['sym','date','stop_pct','R','ret','why']].to_string(index=False))
