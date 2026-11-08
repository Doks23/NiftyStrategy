"""Step 4: render the self-contained HTML report from the template + analysis output."""
import json
from .common import ROOT

KEEP = ["stock","industry","calc_return","low_date","high_date","bo_date","nh_date","days_low_to_bo",
        "gained_before_bo","bo_to_high","nh_to_high","nh_gained_from_low","nh_vol_x","bo_vol_x","bo_gap",
        "drawdown_into_low","rs_gain_low_to_bo","held_higher_low","updown_vol","median_turnover_cr",
        "earn_type","checks","score","tier","base_start","base_hi","base_lo","earnings"]


def run(cfg):
    y = cfg["year"]; cache = cfg["cache"]
    a = json.loads((cache / "analysis.json").read_text())
    c = json.loads((cache / "charts.json").read_text())
    notes_path = ROOT / cfg["notes"]
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    A = [{k: r.get(k) for k in KEEP} for r in sorted(a, key=lambda r: -r["calc_return"])]
    for r in A:
        r["earnings"] = [{k: e[k] for k in ("q","rd","sales_yoy","np_yoy")} for e in r["earnings"]]
    missing = [r["stock"] for r in A if r["stock"] not in notes]
    if missing:
        print(f"  no read note for: {', '.join(missing)} (add them to {cfg['notes']})")
    j = lambda o: json.dumps(o, separators=(",", ":"))
    t = (ROOT / "pipeline" / "template.html").read_text()
    for k, v in {"__DATA__": j(A), "__CHARTS__": j(c), "__READ__": j(notes), "__YEAR__": str(y),
                 "__PYEAR__": str(y - 1), "__IXLO_FROM__": cfg["index_low_window"][0],
                 "__IXLO_TO__": cfg["index_low_window"][1]}.items():
        t = t.replace(k, v)
    # fragment: page content only (the claude.ai Artifact host adds the document skeleton)
    (cache / "page_fragment.html").write_text(t)
    out = ROOT / "reports" / f"winners_{y}.html"
    out.write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                   '</head><body style="margin:0">\n' + t + '\n</body></html>\n')
    print(f"  wrote {out.relative_to(ROOT.parent)} ({len(t)//1024} KB)")
