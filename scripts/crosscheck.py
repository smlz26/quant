"""多源交叉校验。

目的：不是「数据源越多越好」，而是让两个独立来源互相证明。
任何一处对不上，宁可不交易，也不能用错数据。

校验项：
  1. 股票池：交易所官方名单 vs 新浪行情覆盖
  2. 股本：交易所「流通股本」× 价格 是否等于 新浪「流通市值」
  3. 价格：新浪快照 收盘价 vs 腾讯 K 线 收盘价
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aq.data import exchange, tencent  # noqa: E402

SNAP = ROOT / "data" / "snapshots" / "snapshot_20261021.csv"


def load_sina() -> pd.DataFrame:
    if not SNAP.exists():
        raise SystemExit(f"找不到快照 {SNAP}，先跑 scripts/market_pulse.py")
    df = pd.read_csv(SNAP, dtype={"code": str})
    for col in ("price", "float_cap", "mkt_cap", "volume", "amount", "turnover"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def check_universe(sina: pd.DataFrame, ex: pd.DataFrame) -> None:
    print("=" * 78)
    print("校验 1｜股票池覆盖")
    print("=" * 78)
    s_codes = set(sina["code"])
    e_codes = set(ex["code"])

    only_sina = s_codes - e_codes
    only_ex = e_codes - s_codes

    bj = {c for c in only_sina if c.startswith(("8", "4", "9"))}
    print(f"新浪快照          {len(s_codes):>5d} 只")
    print(f"交易所官方(沪深)   {len(e_codes):>5d} 只")
    print(f"仅在交易所（行情站漏）{len(only_ex):>5d} 只")
    print(f"仅在新浪（交易所外） {len(only_sina):>5d} 只，其中北交所 {len(bj)} 只")

    if only_ex:
        sample = ex[ex["code"].isin(list(only_ex))].head(8)
        print("\n  ⚠️ 交易所里有、新浪没有的股票（示例）：")
        for _, r in sample.iterrows():
            print(f"     {r['code']}  {r['name']}  上市 {r['list_date']}  {r['board']}")
        print("     → 需要逐个确认是「停牌」「已退市」还是「行情站遗漏」")

    # 退市股
    delisted = ex[ex["delist_date"].notna()]
    print(f"\n交易所标记已退市：{len(delisted)} 只")
    if not delisted.empty:
        print("  → 这些必须进「已退市股票池」，否则回测存在幸存者偏差")
    print()


def check_shares(sina: pd.DataFrame, ex: pd.DataFrame) -> None:
    print("=" * 78)
    print("校验 2｜股本一致性（交易所流通股本 × 收盘价 ≈ 新浪流通市值）")
    print("=" * 78)
    m = sina.merge(ex[["code", "name", "float_shares", "total_shares", "board"]],
                   on="code", how="inner", suffixes=("", "_ex"))
    for col in ("float_shares", "total_shares", "price", "float_cap"):
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m = m[(m["float_shares"] > 0) & (m["price"] > 0) & m["float_cap"].notna()]

    m["implied_price"] = m["float_cap"] / m["float_shares"]
    m["dev"] = (m["implied_price"] / m["price"] - 1).abs()

    total = len(m)
    bad = m[m["dev"] > 0.01]
    print(f"可比对样本        {total:>5d} 只")
    print(f"偏差 > 1%         {len(bad):>5d} 只  ({len(bad)/total*100:.2f}%)")
    print(f"偏差中位数        {m['dev'].median()*100:.3f}%")
    print(f"偏差 95 分位      {m['dev'].quantile(0.95)*100:.3f}%")

    if not bad.empty:
        print("\n  偏差最大的 8 只：")
        print(f"    {'代码':<8}{'名称':<12}{'新浪价':>9}{'反推价':>9}{'偏差':>9}")
        for _, r in bad.nlargest(8, "dev").iterrows():
            print(f"    {r['code']:<8}{str(r['name'])[:10]:<12}"
                  f"{r['price']:>9.2f}{r['implied_price']:>9.2f}{r['dev']*100:>8.2f}%")
        print("    → 可能是「股本已变更但行情站未更新」，或「停牌期间的 stale 市值」")
    print()


def to_symbol(code: str) -> str:
    """A 股代码 -> 腾讯行情代码。北交所是 bj 前缀，不能用 9 开头判断。"""
    if code.startswith("6"):
        return "sh" + code
    if code.startswith(("0", "3")):
        return "sz" + code
    if code.startswith(("4", "8", "9")):
        return "bj" + code
    return "sh" + code


def check_prices(sina: pd.DataFrame, n: int = 40) -> None:
    print("=" * 78)
    print("校验 3｜价格一致性（新浪快照 vs 腾讯 K 线）")
    print("=" * 78)
    pool = sina[(sina["volume"] > 0) & sina["code"].str.startswith(("0", "3", "6"))]
    if pool.empty:
        print("  没有可比较的样本")
        return
    step = max(1, len(pool) // n)
    sample = pool.iloc[::step].head(n)

    rows = []
    for _, r in sample.iterrows():
        code = r["code"]
        try:
            k = tencent.fetch_kline(to_symbol(code), count=5)
        except Exception:  # noqa: BLE001
            continue
        if k.empty:
            continue
        rows.append({
            "code": code, "name": r["name"],
            "sina": float(r["price"]), "tencent": float(k["close"].iloc[-1]),
            "tx_date": k["date"].iloc[-1].date(),
        })

    cmp_df = pd.DataFrame(rows)
    if cmp_df.empty:
        print("  没有可比较的样本")
        return

    cmp_df["dev"] = (cmp_df["tencent"] / cmp_df["sina"] - 1).abs()
    bad = cmp_df[cmp_df["dev"] > 0.001]

    print(f"抽样              {len(cmp_df)} 只")
    print(f"偏差 > 0.1%       {len(bad)} 只")
    print(f"最大偏差          {cmp_df['dev'].max()*100:.4f}%")
    print(f"腾讯数据截止日    {cmp_df['tx_date'].mode().iloc[0]}")
    if not bad.empty:
        print("\n  不一致的：")
        for _, r in bad.iterrows():
            print(f"    {r['code']} {str(r['name'])[:8]:<9} 新浪 {r['sina']:>8.2f} "
                  f"腾讯 {r['tencent']:>8.2f}  {r['dev']*100:.3f}%")
    else:
        print("\n  ✅ 两个价格源完全一致")
    print()


def main() -> None:
    print("拉取交易所官方主档 ...")
    ex = exchange.fetch_all()
    print(f"  沪深合计 {len(ex)} 只\n")

    sina = load_sina()
    check_universe(sina, ex)
    check_shares(sina, ex)
    check_prices(sina)

    out = ROOT / "reports" / "数据质量校验_20261021.md"
    print("=" * 78)
    print(f"完整校验报告建议每日跑一次，写入 {out.name}")


if __name__ == "__main__":
    main()
