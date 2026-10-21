"""市场情绪快照 v0.2

数据源：新浪（全 A 横截面）+ 腾讯（指数历史）
输出：reports/市场情绪日报_<日期>.md，并归档当日快照
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aq.data import sina, tencent  # noqa: E402
from aq.data.em import board_of, limit_pct  # noqa: E402

REPORT_DIR = ROOT / "reports"
SNAP_DIR = ROOT / "data" / "snapshots"

INDEXES = {
    "上证指数": ("sh000001", "sh000001"),
    "深证综指": ("sz399106", "sz399106"),
    "沪深300": ("sh000300", "sh000300"),
    "中证500": ("sh000905", "sh000905"),
    "中证1000": ("sh000852", "sh000852"),
    "创业板指": ("sz399006", "sz399006"),
}


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if pd.isna(x):
        return 50.0
    return max(lo, min(hi, x))


def pct_rank(history: pd.Series, value: float) -> float:
    s = history.dropna()
    if s.empty or pd.isna(value):
        return float("nan")
    return float((s < value).mean() * 100)


def compute_breadth(snap: pd.DataFrame) -> dict:
    df = snap.copy()
    df = df[df["price"].notna() & (df["price"] > 0) & (df["volume"].fillna(0) > 0)]

    limits = df.apply(lambda r: limit_pct(r["code"], r["name"]), axis=1)
    up_price = (df["pre_close"] * (1 + limits / 100)).round(2)
    down_price = (df["pre_close"] * (1 - limits / 100)).round(2)
    is_up = df["price"] >= up_price - 0.011
    is_down = df["price"] <= down_price + 0.011

    total = len(df)
    up = int((df["pct"] > 0).sum())
    down = int((df["pct"] < 0).sum())

    return {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "total": total, "up": up, "down": down,
        "flat": int((df["pct"] == 0).sum()),
        "up_ratio": up / total * 100 if total else float("nan"),
        "limit_up": int(is_up.sum()), "limit_down": int(is_down.sum()),
        "amount_yi": float(df["amount"].sum()) / 1e8,
        "turnover_median": float(df["turnover"].median()),
        "turnover_gt5": float((df["turnover"] > 5).mean() * 100),
        "pct_median": float(df["pct"].median()),
        "board": df["code"].map(board_of),
        "df": df, "up_flags": is_up, "down_flags": is_down,
    }


def index_context() -> dict:
    sh = tencent.fetch_kline("sh000001", count=800)
    sz = tencent.fetch_kline("sz399106", count=800)
    hs = tencent.fetch_kline("sh000300", count=800)

    m = sh[["date", "volume"]].merge(sz[["date", "volume"]], on="date", suffixes=("_sh", "_sz"))
    m["vol_total"] = m["volume_sh"] + m["volume_sz"]
    today_vol = float(m["vol_total"].iloc[-1])
    vol_pct = pct_rank(m["vol_total"], today_vol)

    hs["ma200"] = hs["close"].rolling(200).mean()
    last = hs.iloc[-1]
    dev = (last["close"] / last["ma200"] - 1) * 100

    return {
        "vol_today_yi": today_vol / 1e4,  # 手 -> 万手
        "vol_pct": vol_pct,
        "hs300_close": float(last["close"]),
        "hs300_ma200": float(last["ma200"]),
        "hs300_dev": float(dev),
        "history_days": len(m),
    }


def score_components(b: dict, ctx: dict) -> list[dict]:
    comps = []

    s1 = ctx["vol_pct"]
    comps.append({
        "name": "两市成交量分位", "raw": f"{ctx['vol_today_yi']:.0f} 万手",
        "score": clamp(s1), "weight": 0.30, "has_history": True,
        "note": f"近 {ctx['history_days']} 个交易日第 {s1:.0f} 百分位（作成交额代理）",
    })

    lu, ld = b["limit_up"], b["limit_down"]
    s2 = 50.0 if (lu + ld) == 0 else clamp(lu / (lu + ld) * 100)
    comps.append({
        "name": "涨停/跌停比", "raw": f"涨停 {lu} / 跌停 {ld}",
        "score": s2, "weight": 0.25, "has_history": False,
        "note": "绝对阈值（历史分位待积累）",
    })

    s3 = clamp(b["up_ratio"])
    comps.append({
        "name": "上涨家数占比", "raw": f"{b['up']} 涨 / {b['down']} 跌（{b['up_ratio']:.1f}%）",
        "score": s3, "weight": 0.20, "has_history": False,
        "note": "绝对阈值（历史分位待积累）",
    })

    dev = ctx["hs300_dev"]
    comps.append({
        "name": "沪深300 偏离 200 日线", "raw": f"{dev:+.2f}%",
        "score": clamp(50 + dev * 5), "weight": 0.15, "has_history": True,
        "note": f"MA200 = {ctx['hs300_ma200']:.2f}",
    })

    s5 = clamp(b["turnover_gt5"] / 30 * 100)
    comps.append({
        "name": "换手率 >5% 个股占比", "raw": f"{b['turnover_gt5']:.1f}%",
        "score": s5, "weight": 0.10, "has_history": False,
        "note": "绝对阈值，30% 记满分",
    })

    return comps


def position_advice(temp: float) -> tuple[str, int]:
    if temp >= 85:
        return "极端过热（亢奋）", 30
    if temp >= 70:
        return "偏热", 50
    if temp >= 55:
        return "温和偏暖", 70
    if temp >= 40:
        return "中性", 80
    if temp >= 25:
        return "偏冷（恐慌初期）", 90
    return "极端恐慌", 100


def build_report(b, ctx, comps, temp, state, target) -> str:
    L = []
    L.append(f"# 市场情绪日报 · {b['date']}\n")
    L.append(f"> 数据源：新浪财经（全 A 横截面）+ 腾讯行情（指数历史）｜"
             f"生成 {datetime.now():%Y-%m-%d %H:%M}\n")

    bar = int(round(temp / 100 * 40))
    L.append("## 一、情绪温度\n")
    L.append(f"**{temp:.1f} / 100** ｜ **{state}** ｜ 建议仓位天花板 **{target}%**\n")
    L.append(f"`{'█' * bar}{'░' * (40 - bar)}`\n")
    L.append("> 温度越高 = 越狂热。**本模型逆向使用：越热越减仓，越冷越加仓。**\n")

    L.append("## 二、分项拆解\n")
    L.append("| 指标 | 当前值 | 得分 | 权重 | 说明 |")
    L.append("|---|---|---|---|---|")
    for c in comps:
        flag = "" if c["has_history"] else " ⚠️"
        L.append(f"| {c['name']} | {c['raw']} | {c['score']:.0f} | "
                 f"{c['weight']*100:.0f}% | {c['note']}{flag} |")
    L.append("\n⚠️ = 暂无历史分位，用绝对阈值，精度有限。\n")

    L.append("## 三、今日盘面\n")
    L.append("| 项目 | 数值 |")
    L.append("|---|---|")
    L.append(f"| 交易个股数 | {b['total']} |")
    L.append(f"| 上涨 / 下跌 / 平盘 | {b['up']} / {b['down']} / {b['flat']} |")
    L.append(f"| 涨停 / 跌停 | {b['limit_up']} / {b['limit_down']} |")
    L.append(f"| 全市场成交额 | {b['amount_yi']:.0f} 亿元 |")
    L.append(f"| 涨跌幅中位数 | {b['pct_median']:+.2f}% |")
    L.append(f"| 换手率中位数 | {b['turnover_median']:.2f}% |")
    L.append(f"| 换手率>5% 占比 | {b['turnover_gt5']:.1f}% |")
    L.append(f"| 沪深300 | {ctx['hs300_close']:.2f}（MA200 偏离 {ctx['hs300_dev']:+.2f}%）|")
    L.append("")

    df = b["df"]
    up_df = df[b["up_flags"]]
    if not up_df.empty:
        top = up_df["name"].value_counts().head(3)  # placeholder
        ind = df["code"].map(board_of).loc[up_df.index]
        L.append("## 四、涨停股板块分布\n")
        L.append("| 板块 | 涨停家数 |")
        L.append("|---|---|")
        for k, v in ind.value_counts().items():
            L.append(f"| {k} | {v} |")
        L.append("")

    L.append("## 五、板块表现\n")
    L.append("| 板块 | 家数 | 涨跌幅中位数 | 涨停数 |")
    L.append("|---|---|---|---|")
    for name, grp in df.groupby(b["board"]):
        mask = b["up_flags"].loc[grp.index]
        L.append(f"| {name} | {len(grp)} | {grp['pct'].median():+.2f}% | {int(mask.sum())} |")
    L.append("")

    L.append("---\n")
    L.append("**v0.2 局限**：涨停比、上涨占比、高换手占比三项尚无历史分位，"
             "按经验阈值打分；成交量为成交额的代理变量。"
             "等每日快照积累满 1 年，全部切换为滚动分位。"
             "**当前请当定性参考，不要直接按仓位数字操作。**\n")
    return "\n".join(L)


def main() -> None:
    print("[1/3] 拉取全 A 快照（新浪）...")
    snap = sina.fetch_snapshot()
    print(f"      {len(snap)} 只")

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y%m%d")
    snap.to_csv(SNAP_DIR / f"snapshot_{today}.csv", index=False, encoding="utf-8-sig")

    print("[2/3] 计算横截面指标 ...")
    b = compute_breadth(snap)

    print("[3/3] 拉取指数历史（腾讯）...")
    ctx = index_context()

    comps = score_components(b, ctx)
    temp = sum(c["score"] * c["weight"] for c in comps)
    state, target = position_advice(temp)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"市场情绪日报_{b['date'].replace('-', '')}.md"
    out.write_text(build_report(b, ctx, comps, temp, state, target), encoding="utf-8")

    print(f"\n情绪温度 = {temp:.1f}/100   {state}   建议仓位上限 {target}%")
    print(f"涨停 {b['limit_up']} / 跌停 {b['limit_down']}   成交额 {b['amount_yi']:.0f} 亿")
    print(f"报告: {out}")


if __name__ == "__main__":
    main()
