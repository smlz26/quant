"""因子有效性诊断：Rank IC。

如果 IC 长期为负，说明因子方向反了或者根本不 work。
这一步是判断「策略差」到底是引擎 bug 还是因子无效的关键。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import backtest_v0 as bt  # noqa: E402


def main() -> None:
    close, volume, open_, meta = bt.load_panel()
    amount = bt.amount_panel(close, volume)
    cap = pd.Series({c: meta["cap"].get(c, np.nan) for c in close.columns}, dtype=float)

    mom20 = close / close.shift(20) - 1
    vol60 = close.pct_change().rolling(60).std()
    turn20 = amount.rolling(20).mean() / cap
    turn_ratio = turn20 / turn20.rolling(60).mean()

    fwd = close.shift(-20) / close - 1          # 未来 20 日收益

    factors = {
        "反转(-20d)": -mom20,
        "低波(-60d std)": -vol60,
        "低换手(-20d)": -turn20,
        "情绪(-换手放大)": -turn_ratio,
    }

    dates = [d for d in close.index if d >= close.index[0] + pd.Timedelta(days=240)]
    dates = dates[::21]  # 约每月

    print(f"{'因子':<18}{'IC均值':>9}{'ICIR':>8}{'t值':>7}{'胜率':>8}{'样本':>7}")
    print("-" * 58)

    for name, f in factors.items():
        ics = []
        for d in dates:
            x = f.loc[d]
            y = fwd.loc[d]
            m = x.notna() & y.notna()
            if m.sum() < 50:
                continue
            ics.append(x[m].rank().corr(y[m].rank()))
        s = pd.Series(ics).dropna()
        if len(s) < 5:
            print(f"{name:<18}{'样本不足':>9}")
            continue
        icir = s.mean() / s.std() if s.std() else 0
        t = icir * np.sqrt(len(s))
        print(f"{name:<18}{s.mean():>9.4f}{icir:>8.2f}{t:>7.2f}"
              f"{(s > 0).mean()*100:>7.0f}%{len(s):>7}")

    # 合成因子
    rows = {}
    for d in dates:
        f_rev = bt.zscore(-mom20.loc[d])
        f_vol = bt.zscore(-vol60.loc[d])
        f_turn = bt.zscore(-turn20.loc[d])
        f_sent = bt.zscore(-turn_ratio.loc[d])
        comp = (0.35 * f_rev + 0.30 * f_vol + 0.15 * f_turn + 0.20 * f_sent)
        comp = bt.neutralize(comp, np.log(cap))
        rows[d] = comp
    comp_df = pd.DataFrame(rows).T

    ics = []
    for d in comp_df.index:
        x, y = comp_df.loc[d], fwd.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= 50:
            ics.append(x[m].rank().corr(y[m].rank()))
    s = pd.Series(ics).dropna()
    icir = s.mean() / s.std() if s.std() else 0
    print("-" * 58)
    print(f"{'合成因子':<18}{s.mean():>9.4f}{icir:>8.2f}"
          f"{icir*np.sqrt(len(s)):>7.2f}{(s > 0).mean()*100:>7.0f}%{len(s):>7}")
    print()
    print("判读：|IC| < 0.02 基本无效；ICIR < 0.3 不稳定；t < 2 不显著。")


if __name__ == "__main__":
    main()

