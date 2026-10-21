"""A 股交易成本模型。

所有参数来自用户中山证券账户 2026-08 ~ 2026-10 的真实交割单反推：

    买入 中际旭创  66,211.92 元 → 费用 11.92 元  ⇒ 万 1.8（且未单独收过户费）
    买入 中国船舶   7,041.00 元 → 费用  5.00 元  ⇒ 触发最低 5 元
    卖出 中文在线  17,675.16 元 → 费用 13.84 元  ⇒ 佣金 5（最低）+ 印花税 8.84
    卖出 长白山    12,957.52 元 → 费用 11.48 元  ⇒ 佣金 5 + 印花税 6.48

关键结论：
    1. 佣金率 万 1.8，最低 5 元/笔 —— 用户明确说「没有免五」
    2. 印花税 0.05%，仅卖出
    3. 过户费未单独列示，已并入佣金（中山证券口径）
    4. 因此最低 5 元的临界金额 = 5 / 0.00018 ≈ 27,778 元
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    """A 股单笔交易成本。

    Args:
        commission_rate: 佣金费率（含过户费的综合费率）
        min_commission: 单笔最低佣金，A 股普遍 5 元
        stamp_duty_rate: 印花税率，仅卖出收取
        transfer_fee_rate: 过户费率，若券商已并入佣金则填 0
    """

    commission_rate: float = 0.00018
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.0

    def fee(self, amount: float, side: str) -> float:
        """单笔费用。side 取 'buy' 或 'sell'；amount 为成交金额（元）。"""
        if amount <= 0:
            return 0.0
        if side not in ("buy", "sell"):
            raise ValueError(f"side 必须是 buy/sell，收到 {side!r}")

        commission = max(amount * self.commission_rate, self.min_commission)
        stamp = amount * self.stamp_duty_rate if side == "sell" else 0.0
        transfer = amount * self.transfer_fee_rate
        return round(commission + stamp + transfer, 2)

    @property
    def breakeven_amount(self) -> float:
        """佣金刚好等于最低收费的成交金额。低于此金额，实际费率高于名义费率。"""
        return self.min_commission / self.commission_rate

    def effective_rate(self, amount: float, side: str) -> float:
        """实际费率（含最低 5 元的影响）。"""
        if amount <= 0:
            return 0.0
        return self.fee(amount, side) / amount

    def round_trip_cost(self, amount: float) -> float:
        """买入 + 卖出的总成本（元）。"""
        return self.fee(amount, "buy") + self.fee(amount, "sell")

    def round_trip_rate(self, amount: float) -> float:
        """买入 + 卖出的综合费率。"""
        if amount <= 0:
            return 0.0
        return self.round_trip_cost(amount) / amount


# 用户账户的默认费率
DEFAULT_FEES = FeeModel()


def annual_cost(capital: float, n_positions: int, rebalances_per_year: int,
                turnover_per_rebalance: float,
                fees: FeeModel = DEFAULT_FEES) -> dict:
    """估算年化交易成本。

    Args:
        capital: 总资金
        n_positions: 持仓数量
        rebalances_per_year: 每年调仓次数（月频 12、双周频 26、周频 52）
        turnover_per_rebalance: 每次调仓的换手比例（0~1）
    """
    per_position = capital / n_positions
    rt_rate = fees.round_trip_rate(per_position)

    # 每次调仓，换手部分承担一次完整买卖成本
    annual_rate = rebalances_per_year * turnover_per_rebalance * rt_rate
    return {
        "per_position": per_position,
        "min_bound_binding": per_position < fees.breakeven_amount,
        "round_trip_rate": rt_rate,
        "annual_rate": annual_rate,
        "annual_cost": capital * annual_rate,
    }
