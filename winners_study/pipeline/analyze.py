"""Step 3: find the move, breakout, 52-week-high reclaim and score each stock's setup.

Definitions (also printed on the page):
- Move      = largest rise from a low in the study year to a later high in that year.
- Breakout  = first close after the low above the prior 50-session high on >=1.5x 50-day avg volume
              (fallback: first close above the prior 50-session high).
- Reclaim   = first close above the prior 250-session high.
- RS        = stock close / benchmark index close.
"""
import json
import numpy as np, pandas as pd


def _jsonable(o):
    """numpy scalars -> plain Python, so JSON gets true/false and numbers, never strings."""
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return str(o)
from .common import read_inputs

MON={"Mar":3,"Jun":6,"Sep":9,"Dec":12}
def qdate(lbl):
    m,y=lbl.split();return pd.Timestamp(int(y),MON[m],1)+pd.offsets.MonthEnd(0)
def result_date(qe):  # estimated announcement: SEBI deadline 45d (60d for Mar quarter)
    return qe+pd.Timedelta(days=60 if qe.month==3 else 45)
def yoy(vals,q,i):
    try:
        lbl=q[i];m,y=lbl.split();prev=f"{m} {int(y)-1}"
        j=q.index(prev);a,b=vals[i],vals[j]
        if a is None or b is None: return None
        if b<=0: return None if a<=0 else 999.0
        return round((a/b-1)*100,1)
    except Exception: return None


def score(a):
    for r in a:
        ref=pd.Timestamp(r["nh_date"] or r["bo_date"])+pd.Timedelta(days=30)
        seen=[e for e in r["earnings"] if pd.Timestamp(r["low_date"])-pd.Timedelta(days=120)<=pd.Timestamp(e["rd"])<=ref and e["np_yoy"] is not None][-3:]
        acc=sum(1 for e in seen if e["np_yoy"]>=25 and (e["sales_yoy"] or 0)>=10)
        turn=any(e["np_yoy"]==999 for e in seen)
        r["earn_seen"]=seen
        r["earn_type"]="Accelerating" if acc>=2 else ("Turnaround" if turn or acc==1 else "Weak / none")
        chk={
         "washout":r["drawdown_into_low"]>=30,
         "rs_thrust":r["rs_gain_low_to_bo"]>=15,
         "higher_low":r["held_higher_low"],
         "bo_volume":r["bo_vol_x"]>=2,
         "accumulation":r["updown_vol"]>1,
         "nh_trend":r.get("nh_trend") is True,
         "rs_new_high":r.get("nh_rs_hi") is True,
         "earnings":r["earn_type"]!="Weak / none",
         "liquidity":r["median_turnover_cr"]>=1}
        chk={k:(None if v is None else bool(v)) for k,v in chk.items()}   # plain bools: numpy bools would serialise as "True"
        r["checks"]=chk;r["score"]=sum(1 for v in chk.values() if v)
    return a


def run(cfg):
    Y=cfg["year"];cache=cfg["cache"];rw=cfg["retest_window"];pw=cfg["prior_low_window"]
    w=read_inputs(cfg);scr=json.loads((cache/"screener.json").read_text())
    IX=pd.read_csv(cache/"dhan"/"INDEX.csv",parse_dates=["date"]).drop_duplicates("date",keep="last").set_index("date")
    ix=IX.close;ixc=ix.pct_change()*100
    out=[];charts={}
    for _,row in w.iterrows():
        s=row["Stock Name"];d=pd.read_csv(cache/"dhan"/f"{s}.csv",parse_dates=["date"]).set_index("date")
        d=d[~d.index.duplicated()]
        d["sma50"]=d.close.rolling(50).mean();d["sma150"]=d.close.rolling(150).mean();d["sma200"]=d.close.rolling(200).mean()
        d["v50"]=d.volume.rolling(50).mean();d["tv"]=d.close*d.volume
        rs=(d.close/ix.reindex(d.index).ffill());d["rs"]=rs/rs.dropna().iloc[0]*100
        y=d.loc[str(Y)]
        # max low->subsequent high
        runmin=y.low.cummin();ratio=y.high/runmin;H=ratio.idxmax();L=y.loc[:H,"low"].idxmin()
        post_ipo=len(d.loc[:L])<120  # too little history before the low for a 200-day average
        mpr=round((y.high[H]/y.low[L]-1)*100,1)
        # breakout: first close after L above prior 50-session high, with vol >= 1.5x v50
        after=d.loc[L:H];bo=None
        for t in after.index[1:]:
            i=d.index.get_loc(t)
            if i<55: continue
            prior=d.iloc[i-50:i]
            if d.close.iloc[i]>prior.high.max() and d.volume.iloc[i]>=1.5*d.v50.iloc[i-1]:
                bo=t;break
        if bo is None:  # fallback: first close above prior 50d high
            for t in after.index[1:]:
                i=d.index.get_loc(t)
                if i>=55 and d.close.iloc[i]>d.iloc[i-50:i].high.max(): bo=t;break
        i=d.index.get_loc(bo);pre=d.iloc[max(0,i-60):i]
        base_hi,base_lo=pre.high.max(),pre.low.min();depth=round((base_hi/base_lo-1)*100,1)
        n3=len(pre)//3;thirds=[pre.iloc[:n3],pre.iloc[n3:2*n3],pre.iloc[2*n3:]];tr=[round((x.high.max()/x.low.min()-1)*100,1) for x in thirds]
        vdry=round(d.volume.iloc[i-10:i].mean()/d.v50.iloc[i-10],2) if i>=60 else None
        bovol=round(d.volume.iloc[i]/d.v50.iloc[i-1],1)
        gap=round((d.open.iloc[i]/d.close.iloc[i-1]-1)*100,1);boday=round((d.close.iloc[i]/d.close.iloc[i-1]-1)*100,1)
        w52=d.iloc[max(0,i-250):i]
        pct_from_hi=round((d.close.iloc[i]/w52.high.max()-1)*100,1);pct_above_lo=round((d.close.iloc[i]/w52.low.min()-1)*100,1)
        c=d.iloc[i]
        trend=bool(c.close>c.sma50>c.sma150>c.sma200) if not np.isnan(c.sma200) else None
        sma200_rising=bool(d.sma200.iloc[i]>d.sma200.iloc[i-20]) if i>=220 else None
        rsw=d.rs.iloc[max(0,i-250):i+1];rs_hi=bool(d.rs.iloc[i]>=rsw.max()*0.98)
        # RS made new 52w high before price did (in 20 days before breakout)
        rs_lead=bool(any(d.rs.iloc[k]>=d.rs.iloc[max(0,k-250):k].max() for k in range(max(1,i-20),i)))
        p50=d.iloc[max(0,i-50):i];chg=p50.close.diff()
        ud=round(p50.volume[chg>0].sum()/max(1,p50.volume[chg<0].sum()),2)
        pre_decline=round((d.loc[:L].iloc[-250:].high.max()/d.low[L]-1)*100,1)  # drop into low
        pre_dd=round((1-d.low[L]/d.loc[:L].iloc[-250:].high.max())*100,1)
        gained_before_bo=round((c.close/d.low[L]-1)*100,1);after_bo=round((y.high[H]/c.close-1)*100,1)
        ix_ret=round((ix.reindex([H]).ffill().iloc[0]/ix.reindex([L]).ffill().iloc[0]-1)*100,1) if L in ix.index and H in ix.index else None
        medtv=round(pre.tv.median()/1e7,2)  # crore
        # earnings
        sc=scr.get(s) or {};q=sc.get("q",[]);eq=[]
        for k,lbl in enumerate(q):
            try: qe=qdate(lbl)
            except Exception: continue
            eq.append({"q":lbl,"rd":str(result_date(qe).date()),"sales":sc["sales"][k] if sc.get("sales") else None,"np":sc["np"][k] if sc.get("np") else None,
                       "eps":sc["eps"][k] if sc.get("eps") else None,"sales_yoy":yoy(sc.get("sales") or [],q,k),"np_yoy":yoy(sc.get("np") or [],q,k)})
        known=[e for e in eq if pd.Timestamp(e["rd"])<=bo]
        last2=known[-2:]
        near=[e for e in eq if abs((pd.Timestamp(e["rd"])-bo).days)<=25]
        # archetype
        days_L_bo=(bo-L).days
        if post_ipo: arch="Post-IPO base"
        elif medtv<0.5: arch="Illiquid / thin float"
        elif gap>=5 and near: arch="Earnings gap"
        elif pre_dd>=40 and days_L_bo<=45: arch="V-reversal / turnaround"
        elif depth<=40 and (vdry or 9)<=0.9: arch="Base breakout (tight)"
        else: arch="Base breakout (loose)"
        rec=dict(stock=s,industry=row["Basic Industry"],csv_return=row.get(cfg["return_col"]),calc_return=mpr,
          low_date=str(L.date()),high_date=str(H.date()),bo_date=str(bo.date()),days_low_to_bo=days_L_bo,days_bo_to_high=(H-bo).days,
          gained_before_bo=gained_before_bo,bo_to_high=after_bo,index_ret=ix_ret,
          base_depth=depth,base_thirds=tr,vol_dryup=vdry,bo_vol_x=bovol,bo_gap=gap,bo_day=boday,
          pct_from_52w_hi=pct_from_hi,pct_above_52w_lo=pct_above_lo,trend_template=trend,sma200_rising=sma200_rising,
          rs_at_high=rs_hi,rs_led=rs_lead,updown_vol=ud,drawdown_into_low=pre_dd,median_turnover_cr=medtv,
          archetype=arch,earnings_before_bo=last2,earnings_near_bo=near,earnings=eq,
          about=(sc.get("about") or "")[:400],ratios=sc.get("ratios",{}),base_start=str(pre.index[0].date()),base_hi=base_hi,base_lo=base_lo,
          csv_eps=row.get("Avg QoQ EPS Growth(%)"),csv_sales=row.get("Avg QoQ Sales Growth(%)"),csv_voldays=row.get("Days with buying volume > 50-day MA"),
          tier=1 if mpr>=cfg.get("tier1_min",200) else 2,source=row["_file"])
        # 52w-high reclaim after the low
        nh=None
        for t in d.loc[L:H].index[1:]:
            k=d.index.get_loc(t)
            if k>=200 and d.close.iloc[k]>d.iloc[max(0,k-250):k].high.max(): nh=t;break
        rec["nh_date"]=str(nh.date()) if nh is not None else None
        rec["nh_to_high"]=round((y.high[H]/d.close[nh]-1)*100,1) if nh is not None else None
        rec["nh_gained_from_low"]=round((d.close[nh]/d.low[L]-1)*100,1) if nh is not None else None
        rec["nh_vol_x"]=round(d.volume[nh]/d.v50.shift(1)[nh],1) if nh is not None else None
        if nh is not None:
            k=d.index.get_loc(nh);p2=d.iloc[max(0,k-40):k];rec["nh_base_depth"]=round((p2.high.max()/p2.low.min()-1)*100,1)
            rec["nh_vdry"]=round(d.volume.iloc[k-10:k].mean()/d.v50.iloc[k-10],2)
            rec["nh_trend"]=bool(d.close.iloc[k]>d.sma50.iloc[k]>d.sma200.iloc[k]) if not pd.isna(d.sma200.iloc[k]) else None
            rec["nh_rs_hi"]=bool(d.rs.iloc[k]>=d.rs.iloc[max(0,k-250):k].max())
            kn=[e for e in eq if pd.Timestamp(e["rd"])<=nh];rec["earn_at_nh"]=kn[-1] if kn else None
            rec["earn_near_nh"]=[e["q"] for e in eq if -10<=(nh-pd.Timestamp(e["rd"])).days<=30]
        # index retest: stock low inside the index retest window vs its earlier low that year
        a=d.loc[rw[0]:rw[1]].low.min();b=d.loc[pw[0]:pw[1]].low.min()
        rec["held_higher_low"]=bool(a>b*1.02) if not post_ipo and pd.notna(a) and pd.notna(b) else None
        rec["rs_gain_low_to_bo"]=round(rec["gained_before_bo"]-(ix.asof(bo)/ix.asof(L)-1)*100,1)
        rec["index_ret"]=round((ix.asof(H)/ix.asof(L)-1)*100,1)
        out.append(rec)
        cs=d.loc[max(d.index[0],bo-pd.Timedelta(days=300)):min(d.index[-1],H+pd.Timedelta(days=75))]
        r2=lambda v:None if pd.isna(v) else round(float(v),2)
        charts[s]=[[str(t.date()),r2(x.open),r2(x.high),r2(x.low),r2(x.close),int(x.volume),r2(x.sma50),r2(x.sma200),r2(x.v50),r2(x.rs)] for t,x in cs.iterrows()]
        # benchmark columns for the page: close, daily %, open, high, low
        for row_ in charts[s]:
            t_=pd.Timestamp(row_[0]);v_=ix.asof(t_);ch_=ixc.get(t_)
            row_.append(round(float(v_),2) if pd.notna(v_) else None)
            row_.append(round(float(ch_),2) if ch_ is not None and pd.notna(ch_) else None)
            if t_ in IX.index:
                o_=IX.loc[t_];row_.extend([round(float(o_.open),2),round(float(o_.high),2),round(float(o_.low),2)])
            else: row_.extend([None,None,None])
        print(f"  {s}: move {mpr}% (file {rec['csv_return']}%), low {rec['low_date']}, breakout {rec['bo_date']}, reclaim {rec['nh_date']}")
    out=score(out)
    (cache/"analysis.json").write_text(json.dumps(out,default=_jsonable))
    (cache/"charts.json").write_text(json.dumps(charts))
