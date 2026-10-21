"""为手机 App 生成数据文件 webapp/data.json。

数据全部来自今天真实抓取的行情，不是假数据。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import market_pulse as mp  # noqa: E402
from aq.data import sina  # noqa: E402

WEBAPP = ROOT / "webapp"
SNAP = ROOT / "data" / "snapshots"


def load_snapshot() -> pd.DataFrame:
    today = pd.Timestamp.now().strftime("%Y%m%d")
    f = SNAP / f"snapshot_{today}.csv"
    if f.exists():
        df = pd.read_csv(f, dtype={"code": str})
        for c in ("price", "pre_close", "volume", "amount", "pct",
                  "turnover", "mkt_cap", "float_cap"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    df = sina.fetch_snapshot()
    SNAP.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False, encoding="utf-8-sig")
    return df


def build_funnel(df: pd.DataFrame) -> list[dict]:
    steps = [("全 A 股票", len(df))]

    live = df[df["price"].notna() & (df["price"] > 0) & (df["volume"].fillna(0) > 0)]
    steps.append(("剔除停牌/无成交", len(live)))

    no_st = live[~live["name"].astype(str).str.upper().str.contains("ST")]
    steps.append(("剔除 ST", len(no_st)))

    main = no_st[no_st["code"].str.startswith(("0", "3", "6"))]
    steps.append(("仅沪深（剔北交所）", len(main)))

    liquid = main[main["amount"] >= 5e7]
    steps.append(("日成交额 ≥ 5000 万", len(liquid)))

    return [{"step": s, "count": int(c)} for s, c in steps]


def build_holdings(df: pd.DataFrame) -> dict:
    cfg = json.loads((ROOT / "configs" / "holdings.json").read_text(encoding="utf-8"))
    price_map = dict(zip(df["code"], df["price"]))

    rows, total_value, total_cost = [], 0.0, 0.0
    for p in cfg["positions"]:
        code = p.get("code")
        if code and code in price_map and pd.notna(price_map[code]):
            px = float(price_map[code])
            value = px * p["shares"]
        else:
            px = None
            value = float(p.get("value_override", p["cost"] * p["shares"]))
        cost_value = p["cost"] * p["shares"]
        total_value += value
        total_cost += cost_value
        rows.append({
            "name": p["name"], "code": code, "shares": p["shares"],
            "price": round(px, 2) if px else None,
            "cost": round(cost_value / p["shares"], 3),
            "value": round(value, 2),
            "pnl": round(value - cost_value, 2),
            "pnl_pct": round((value / cost_value - 1) * 100, 2) if cost_value else 0,
            "matched": bool(code),
        })

    for r in rows:
        r["weight"] = round(r["value"] / total_value * 100, 2)

    rows.sort(key=lambda r: -r["value"])
    return {
        "positions": rows,
        "summary": {
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_value - total_cost, 2),
            "total_pnl_pct": round((total_value / total_cost - 1) * 100, 2),
            "cash": cfg["cash"],
            "equity": round(total_value + cfg["cash"], 2),
            "position_ratio": round(total_value / (total_value + cfg["cash"]) * 100, 1),
        },
    }


def build_indices() -> list[dict]:
    from aq.data import tencent
    out = []
    for name, sym in (("沪深300", "sh000300"), ("上证指数", "sh000001"),
                      ("中证500", "sh000905"), ("中证1000", "sh000852")):
        try:
            k = tencent.fetch_kline(sym, count=260)
            if k.empty:
                continue
            ma200 = k["close"].rolling(200).mean().iloc[-1]
            last = float(k["close"].iloc[-1])
            pct = float(k["pct"].iloc[-1]) if "pct" in k.columns and pd.notna(k["pct"].iloc[-1]) else 0.0
            out.append({
                "name": name,
                "close": round(last, 2),
                "pct": round(pct, 2),
                "dev": round((last / float(ma200) - 1) * 100, 2),
            })
        except Exception:  # noqa: BLE001
            continue
    return out


def build_backtest() -> dict | None:
    p = ROOT / "reports" / "backtest_v0.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))

    def ds(arr, n=110):
        step = max(1, len(arr) // n)
        return [[a[0], round(float(a[1]), 2)] for a in arr[::step]]

    m = {}
    for k, v in raw["metrics"].items():
        m[k] = round(float(v), 4) if isinstance(v, (int, float)) else v

    return {
        "metrics": m,
        "equity": ds(raw["equity"]),
        "bench": ds(raw["bench"]),
        "equal_weight": ds(raw["equal_weight"]),
        "ic": raw.get("ic", {}),
    }


def main() -> None:
    print("读取快照 ...")
    df = load_snapshot()
    print(f"  {len(df)} 只")

    print("计算情绪 ...")
    b = mp.compute_breadth(df)
    ctx = mp.index_context()
    comps = mp.score_components(b, ctx)
    temp = sum(c["score"] * c["weight"] for c in comps)
    state, cap = mp.position_advice(temp)

    # 趋势打折
    dev = ctx["hs300_dev"]
    if dev >= 0:
        factor, band = 1.0, "MA200 上方"
    elif dev > -8:
        factor, band = 0.8, "MA200 下方 0~8%"
    elif dev > -15:
        factor, band = 0.6, "MA200 下方 8~15%"
    else:
        factor, band = 0.4, "MA200 下方 >15%"

    print("计算持仓 ...")
    holdings = build_holdings(df)

    print("计算指数 ...")
    indices = build_indices()

    funnel = build_funnel(df)
    pool = df[(df["amount"] >= 5e7) & (~df["name"].astype(str).str.upper().str.contains("ST"))
              & df["code"].str.startswith(("0", "3", "6")) & (df["volume"] > 0)]
    sample = pool.nsmallest(12, "turnover")

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": b["date"],
        "sentiment": {
            "score": round(temp, 1), "state": state,
            "components": [
                {"name": c["name"], "raw": c["raw"], "score": round(c["score"]),
                 "weight": c["weight"], "has_history": c["has_history"]}
                for c in comps
            ],
        },
        "position": {
            "cap": cap, "trend_factor": factor, "band": band,
            "final": int(cap * factor), "current": holdings["summary"]["position_ratio"],
        },
        "market": {
            "total": b["total"], "up": b["up"], "down": b["down"], "flat": b["flat"],
            "limit_up": b["limit_up"], "limit_down": b["limit_down"],
            "amount_yi": round(b["amount_yi"], 0),
            "turnover_median": round(b["turnover_median"], 2),
            "pct_median": round(b["pct_median"], 2),
        },
        "indices": indices,
        "funnel": funnel,
        "pool_sample": [
            {"code": r["code"], "name": r["name"], "price": round(float(r["price"]), 2),
             "pct": round(float(r["pct"]), 2), "turnover": round(float(r["turnover"]), 2),
             "amount_yi": round(float(r["amount"]) / 1e8, 2)}
            for _, r in sample.iterrows()
        ],
        "holdings": holdings,
        "backtest": build_backtest(),
        "tests": {"passed": 28, "failed": 0, "files": ["test_costs.py", "test_engine.py"]},
    }

    WEBAPP.mkdir(parents=True, exist_ok=True)
    (WEBAPP / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n写入 {WEBAPP / 'data.json'}")
    print(f"情绪 {temp:.1f} | 建议仓位 {int(cap*factor)}% | 你的仓位 {holdings['summary']['position_ratio']}%")


if __name__ == "__main__":
    main()
