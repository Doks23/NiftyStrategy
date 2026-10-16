import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/NiftyStrategy")
pn = pd.read_pickle("data/bhav/panel.pkl")
liq = (pn["Turnover"].iloc[-60:].mean() / 1e7) >= 10
syms = [s for s in pn["Close"].columns[liq] if not any(k in s for k in ("ETF","BEES","GOLD","SILVER","LIQUID","NIFTY"))]
def atr(h,l,c,n):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
def run(s, old=False):
    df = pd.DataFrame({k: pn[k][s] for k in ["Open","High","Low","Close","Volume"]}).dropna()
    if len(df) < 120: return 0, 0
    O,H,L,C,V = (df[k] for k in ["Open","High","Low","Close","Volume"])
    e10,e21,e50 = (C.ewm(span=n,adjust=False).mean() for n in (10,21,50))
    aF,aS,vA = atr(H,L,C,5), atr(H,L,C,20), V.rolling(50,min_periods=20).mean()
    hiT,loT = H.rolling(3).max(), L.rolling(3).min()
    upc = (C>C.shift()).astype(int).rolling(10).sum(); roc = (C/C.shift(20)-1)*100
    gap = (O>=H.shift()*1.04)&(V>=2.5*vA)&(C>O)
    ordl = (roc>=25)&(upc>=7)&(C>e10)&(e10>e21)&(e21>e50)
    mom = gap|ordl
    ctest = (aF<=0.75*aS)&((e10-e21).abs()<=0.75*aS)&(((C.rolling(3).max()-C.rolling(3).min())<=0.6*aS)|((H<H.shift())&(L>L.shift())))
    st=0; mh=np.nan; mhb=0; ch=np.nan; cl=np.nan; cb=0; last=-99; n=0
    start = df.index.get_indexer([df.index[df.index>=pd.Timestamp("2025-10-13")][0]])[0]
    for i in range(60,len(df)):
        h,l,c=H.iat[i],L.iat[i],C.iat[i]
        if mom.iat[i]:
            if st==0: st=1
            if np.isnan(mh) or h>mh: mh,mhb=h,i
        elif st>0 and h>mh: mh,mhb=h,i
        if st>0 and i-mhb>40: st,mh,ch=0,np.nan,np.nan
        pb = np.nan if np.isnan(mh) else (mh-l)/mh*100
        if st==1 and pb>=6 and i-mhb>=3: st=2
        if st==2 and (pb>25 or c<e50.iat[i]): st,mh,ch=0,np.nan,np.nan
        con=False
        if st==2 and ctest.iat[i] and i-mhb>=3: con=True; ch,cl,cb=hiT.iat[i],loT.iat[i],i
        if st==2 and not np.isnan(ch) and not con:
            r=h-l
            ok=(i-cb<=5 and c>ch and C.iat[i-1]<=ch and r>=1.3*aF.iat[i-1] and r>0 and (c-l)/r>=0.7
                and V.iat[i]>=1.5*vA.iat[i] and i-last>=10 and c>e10.iat[i] and (c-cl)/c<=0.08)
            if ok:
                last=i; st=1; mh,mhb=h,i
                if i>=start: n+=1
            elif i-cb>5: ch=np.nan
    return n, 1
tot=sum(run(s)[0] for s in syms)
print(f"universe {len(syms)} liquid NSE stocks, 1y: {tot} signals = {tot/len(syms):.2f} per stock/yr, ~{tot/250:.1f} per day across universe")
for s in ["CAMS","BEL","HAL","DIXON","RELIANCE","STLTECH","HINDCOPPER"]:
    if s in pn["Close"]: print(s, run(s)[0])
