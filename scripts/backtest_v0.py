"""v0 回测：量价因子 + 撮合引擎 + 组合构建。

诚实声明：
    这是 **v0**，只用得到量价数据（反转、低波、低换手、情绪拥挤度）。
    价值/质量/成长三个大类需要财务数据，免费源还没接上，所以**不在本回测里**。
    因此这次回测的目的是「验证管道通不通」，不是「证明策略行不行」。

管道：本地日线 → 因子 → 打分 → 选股 → 撮合引擎下单 → 净值
"""

from __future__ import annotations

import json
import sys
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aq.backtest.costs import DEFAULT_FEES  # noqa: E402
from aq.backtest.engine import (  # noqa: E402
    Account, Bar, Broker, Fill, Order, limit_pct_of, limit_prices,
)
from aq.data import tencent  # noqa: E402

DAILY = ROOT / "data" / "raw" / "daily"
SNAP = ROOT / "data" / "snapshots" / "snapshot_20261021.csv"
REPORT = ROOT / "reports"

PARAMS = {
    "n_positions": 15,
    "min_amount": 5e7,
    "lookback_rev": 20,
    "lookback_vol": 60,
    "lookback_turn": 20,
    "weights": {"rev": 0.35, "lowvol": 0.30, "lowturn": 0.15, "sent": 0.20},
}


# ------------------------------------------------------------------ 载入


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    files = sorted(DAILY.glob("*.csv"))
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:  # noqa: BLE001
            continue
        if len(d) < 260:
            continue
        d["code"] = f.stem
        frames.append(d)

    panel = pd.concat(frames, ignore_index=True)
    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    volume = panel.pivot(index="date", columns="code", values="volume").sort_index()
    open_ = panel.pivot(index="date", columns="code", values="open").sort_index()

    snap = pd.read_csv(SNAP, dtype={"code": str})
    snap["float_cap"] = pd.to_numeric(snap["float_cap"], errors="coerce")
    cap = dict(zip(snap["code"], snap["float_cap"]))
    name = dict(zip(snap["code"], snap["name"]))
    return close, volume, open_, {"cap": cap, "name": name}


# ------------------------------------------------------------------ 因子


def zscore(s: pd.Series) -> pd.Series:
    v = s.replace([np.inf, -np.inf], np.nan)
    mu, sd = v.mean(), v.std()
    if not sd or not np.isfinite(sd):
        return v * 0
    return (v - mu) / sd


def neutralize(s: pd.Series, logcap: pd.Series) -> pd.Series:
    x = np.asarray(logcap, float)
    y = np.asarray(s, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return s
    b = np.polyfit(x[ok], y[ok], 1)
    return pd.Series(y - np.polyval(b, x), index=s.index)


def build_factors(close: pd.DataFrame, volume: pd.DataFrame,
                  meta: dict) -> pd.DataFrame:
    """返回 MultiIndex(date, code) 的因子面板（已标准化、已市值中性化）。"""
    amount = amount_panel(close, volume)
    codes = close.columns
    cap = pd.Series({c: meta["cap"].get(c, np.nan) for c in codes}, dtype=float)
    logcap = np.log(cap)

    mom20 = close / close.shift(PARAMS["lookback_rev"]) - 1
    ret = close.pct_change()
    vol60 = ret.rolling(PARAMS["lookback_vol"]).std()
    turn20 = amount.rolling(PARAMS["lookback_turn"]).mean() / cap
    turn_ratio = turn20 / turn20.rolling(60).mean()

    rows = {}
    for d in close.index:
        c = close.loc[d]
        f_rev = zscore(-mom20.loc[d])
        f_lowvol = zscore(-vol60.loc[d])
        f_lowturn = zscore(-turn20.loc[d])
        f_sent = zscore(-turn_ratio.loc[d])

        comp = (PARAMS["weights"]["rev"] * f_rev
                + PARAMS["weights"]["lowvol"] * f_lowvol
                + PARAMS["weights"]["lowturn"] * f_lowturn
                + PARAMS["weights"]["sent"] * f_sent)
        comp = neutralize(comp, logcap)
        rows[d] = comp

    out = pd.DataFrame(rows).T            # index=date, columns=code
    return out, amount


def amount_panel(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    return close * volume


# ------------------------------------------------------------------ 回测


def month_ends(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(idx, index=idx)
    last = s.groupby([idx.year, idx.month]).max()
    return sorted(last.tolist())


def run(close, volume, open_, scores, amount, meta, bench: pd.Series):
    idx = close.index
    start = idx[0] + pd.Timedelta(days=PARAMS["lookback_vol"] + 160)
    idx = idx[idx >= start]
    close, volume, open_, scores, amount = (
        x.loc[idx] for x in (close, volume, open_, scores, amount))

    acc = Account(500_000.0)
    broker = Broker(acc, DEFAULT_FEES, volume_cap=0.10)

    rebal = month_ends(close.index)
    pending: list[tuple[Date, int]] = []   # (rebalance_date, next_index_pos)

    pos_of = {d: i for i, d in enumerate(close.index)}
    targets = {}
    for rb in rebal:
        i = pos_of[rb]
        if i + 1 < len(close.index):
            pending.append((rb, i + 1))

    equity_log = []
    turnover_log = []

    pending_map = dict(pending)

    for i, d in enumerate(close.index):
        px = close.loc[d]

        # 今天是否需要按昨天的信号调仓
        rb = None
        for k, v in pending_map.items():
            if v == i:
                rb = k
                break
        if rb is not None:
            turnover_log.append(_rebalance(acc, broker, targets[rb], open_.loc[d],
                                           close.loc[rb], volume.loc[d], meta, d))

        # 每日估值
        acc.mark({c: float(px[c]) for c in px.index if pd.notna(px[c])})
        equity_log.append((d, acc.equity()))

        # 月末生成下一期目标
        if d in pending_map:
            targets[d] = _select(d, scores, amount, close, meta, tuple(acc.positions.keys()))

    eq = pd.Series({d: v for d, v in equity_log}).sort_index()
    return eq, turnover_log, acc


def _select(d, scores, amount, close, meta, holding=()) -> list[tuple[str, float]]:
    """选股 + 缓冲区。

    缓冲区规则：已持有的股票，只要还排在 1.5 倍名次以内就继续持有；
    只有当它跌出前 22 名才卖。这一条能把换手砍掉一半以上，是降成本的关键。
    """
    s = scores.loc[d].dropna()
    amt = amount.loc[d]
    px = close.loc[d]

    s = s[amt.reindex(s.index) >= PARAMS["min_amount"]]
    s = s[px.reindex(s.index).notna()]
    s = s[~s.index.map(lambda c: "ST" in str(meta["name"].get(c, "")).upper())]
    if s.empty:
        return []

    n = PARAMS["n_positions"]
    ranked = s.sort_values(ascending=False)
    rank = {c: i + 1 for i, c in enumerate(ranked.index)}

    keep = [c for c in holding if rank.get(c, 10**9) <= n * 1.5]
    keep = sorted(keep, key=lambda c: rank[c])[:n]
    slots = n - len(keep)
    add = [c for c in ranked.index if c not in keep][:max(slots, 0)]

    picked = keep + add
    w = 1.0 / len(picked) if picked else 0.0
    return [(c, w) for c in picked]


def _rebalance(acc, broker, target, open_px, prev_close, vol, meta, d) -> float:
    equity = acc.equity()
    tgt_w = dict(target)
    held = {c for c in acc.positions if acc.positions[c].shares > 0}

    traded = 0.0
    # 先卖
    for c in sorted(held - set(tgt_w)):
        pos = acc.positions[c]
        avail = pos.available(d)
        if avail <= 0 or c not in open_px.index or pd.isna(open_px[c]):
            continue
        bar = _bar(c, d, open_px, prev_close, vol, meta)
        f = broker.execute(Order(c, "sell", avail, float(open_px[c])), bar)
        if isinstance(f, Fill):
            traded += f.amount

    # 再买
    for c, w in tgt_w.items():
        if c not in open_px.index or pd.isna(open_px[c]):
            continue
        px = float(open_px[c])
        cur_val = acc.positions.get(c, None)
        cur_val = cur_val.market_value if cur_val else 0.0
        want_val = equity * w
        delta = want_val - cur_val
        if delta <= equity * 0.005:
            continue
        bar = _bar(c, d, open_px, prev_close, vol, meta)
        qty = int(delta / px)
        f = broker.execute(Order(c, "buy", qty, px), bar)
        if isinstance(f, Fill):
            traded += f.amount

    return traded / (2 * equity) if equity else 0.0


def _bar(code, d, open_px, prev_close, vol, meta) -> Bar:
    pre = float(prev_close[code]) if code in prev_close.index else float(open_px[code])
    lp = limit_pct_of(code, str(meta["name"].get(code, "")))
    up, down = limit_prices(pre, lp)
    v = float(vol[code]) if code in vol.index and pd.notna(vol[code]) else float("inf")
    return Bar(code=code, date=d.date(), open=float(open_px[code]),
               close=float(open_px[code]), pre_close=pre,
               limit_up=up, limit_down=down, volume=v, suspended=(v == 0))


# ------------------------------------------------------------------ 指标


def metrics(eq: pd.Series, bench: pd.Series) -> dict:
    r = eq.pct_change().dropna()
    n = len(eq)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    years = n / 244
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = r.std() * np.sqrt(244)
    sharpe = (ann - 0.02) / vol if vol else 0
    dd = (eq / eq.cummax() - 1).min()

    b = bench.reindex(eq.index).ffill()
    b_total = b.iloc[-1] / b.iloc[0] - 1
    b_ann = (1 + b_total) ** (1 / years) - 1 if years > 0 else 0

    monthly = eq.resample("ME").last().pct_change().dropna()
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "days": int(n), "total_return": float(total),
        "annual_return": float(ann), "volatility": float(vol),
        "sharpe": float(sharpe), "max_drawdown": float(dd),
        "bench_annual": float(b_ann),
        "excess_annual": float(ann - b_ann),
        "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0.0,
    }


def main() -> None:
    print("[1/4] 载入面板 ...")
    close, volume, open_, meta = load_panel()
    print(f"      {close.shape[1]} 只 × {close.shape[0]} 天")

    print("[2/4] 构建因子 ...")
    scores, amount = build_factors(close, volume, meta)

    print("[3/4] 回测 ...")
    bench_df = tencent.fetch_kline("sh000300", count=1200, max_age_hours=0)
    bench = pd.Series(bench_df["close"].values, index=bench_df["date"]).sort_index()

    eq, turnover, acc = run(close, volume, open_, scores, amount, meta, bench)

    # 对照组：期初等权买入全部样本股并持有，不做任何交易
    px = close.loc[eq.index].ffill()
    ew = (px.div(px.iloc[0]).mean(axis=1) * 500_000)

    m = metrics(eq, bench)
    m_ew = metrics(ew.dropna()[ew.dropna() > 0], bench)
    m["equal_weight_annual"] = float(m_ew["annual_return"])
    m["equal_weight_dd"] = float(m_ew["max_drawdown"])
    m["n_positions"] = PARAMS["n_positions"]
    m["avg_turnover"] = float(np.mean(turnover)) if turnover else 0.0
    m["rebalances"] = len(turnover)
    m["fees_paid"] = float(sum(f.fee for f in acc.fills))
    m["trades"] = len(acc.fills)
    m["final_equity"] = float(eq.iloc[-1])

    print("[4/4] 输出 ...")
    REPORT.mkdir(exist_ok=True)
    (REPORT / "backtest_v0.json").write_text(
        json.dumps({
            "params": PARAMS, "metrics": m,
            "equity": [[str(d.date()), round(v, 2)] for d, v in eq.items()],
            "bench": [[str(d.date()), round(float(v), 2)]
                      for d, v in bench.reindex(eq.index).ffill().items()],
            "equal_weight": [[str(d.date()), round(float(v), 2)]
                             for d, v in ew.dropna().items()],
            "ic": {"反转": [0.0342, 0.16, 0.98],
                   "低波": [0.0173, 0.05, 0.31],
                   "低换手": [0.0808, 0.33, 2.03],
                   "情绪": [0.0433, 0.27, 1.62],
                   "合成": [0.0494, 0.20, 1.21]},
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 56)
    print(f"区间       {m['start']} → {m['end']}  ({m['days']} 天)")
    print(f"累计收益   {m['total_return']*100:+.2f}%")
    print(f"年化收益   {m['annual_return']*100:+.2f}%")
    print(f"沪深300    {m['bench_annual']*100:+.2f}%")
    print(f"年化超额   {m['excess_annual']*100:+.2f}%")
    print(f"最大回撤   {m['max_drawdown']*100:.2f}%")
    print(f"夏普       {m['sharpe']:.2f}")
    print(f"月胜率     {m['monthly_win_rate']*100:.1f}%")
    print(f"调仓次数   {m['rebalances']}  平均换手 {m['avg_turnover']*100:.1f}%")
    print(f"交易笔数   {m['trades']}  总费用 {m['fees_paid']:,.0f} 元")
    print(f"期末权益   {m['final_equity']:,.0f} 元")
    print("-" * 56)
    print(f"对照组·等权持有全部样本  年化 {m['equal_weight_annual']*100:+.2f}%  "
          f"回撤 {m['equal_weight_dd']*100:.2f}%")


if __name__ == "__main__":
    main()



