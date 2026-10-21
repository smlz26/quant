"""回测撮合内核：账户、持仓、A 股交易规则。

这一层是整个系统的「裁判」。它只做一件事：
    给定一条委托和当天的行情，判断「这笔单子到底能不能成交、成交多少」。

设计原则：
    1. 规则集中在这里，策略层不许绕过
    2. 所有拒绝都要有明确的 reason，方便事后审计
    3. 用「批次（lot）」记录持仓，天然支持 T+1 和红利税的持股期限
    4. 用原始价 + 显式处理分红送转，不用前复权价（前复权会污染涨跌停判断）
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field

from .costs import DEFAULT_FEES, FeeModel

LOT_SIZE = 100


def as_date(x) -> _dt.date:
    """把 pandas.Timestamp / datetime / date 统一成 datetime.date。

    坑：pd.Timestamp 继承自 datetime.datetime，而 datetime 又继承自 date。
    所以必须先判 datetime —— 否则 isinstance(x, date) 为真，Timestamp 被原样放行，
    后面 `Timestamp < date` 就会抛 TypeError。
    """
    if isinstance(x, _dt.datetime):   # 含 pd.Timestamp
        return x.date()
    if isinstance(x, _dt.date):
        return x
    raise TypeError(f"无法识别的日期类型: {type(x)!r}")


# ---------------------------------------------------------------- 行情与委托


@dataclass(frozen=True)
class Bar:
    """一天的行情。价格全部用「原始价」，不做复权。"""

    code: str
    date: date
    open: float
    close: float
    pre_close: float
    limit_up: float
    limit_down: float
    volume: float = math.inf      # 成交量（股）
    suspended: bool = False       # 停牌

    def check_consistency(self) -> None:
        if self.suspended:
            return
        if not (self.limit_down <= self.close <= self.limit_up + 1e-6):
            raise ValueError(
                f"{self.code} {self.date} 收盘价 {self.close} 超出涨跌停 "
                f"[{self.limit_down}, {self.limit_up}]，数据有误"
            )


@dataclass(frozen=True)
class Order:
    code: str
    side: str          # 'buy' / 'sell'
    qty: int           # 委托股数（会被整手规则修正）
    price: float       # 假设的成交价


@dataclass
class Fill:
    code: str
    side: str
    qty: int
    price: float
    amount: float
    fee: float
    date: date
    reason: str = "ok"

    @property
    def cash_delta(self) -> float:
        return -self.amount - self.fee if self.side == "buy" else self.amount - self.fee


@dataclass
class Rejection:
    code: str
    side: str
    qty: int
    reason: str


# ---------------------------------------------------------------- 持仓


@dataclass
class Lot:
    """一批同一天买入的股票。"""

    shares: int
    buy_date: date


@dataclass
class Position:
    code: str
    lots: list[Lot] = field(default_factory=list)
    cost_amount: float = 0.0      # 含买入费用的总成本
    last_price: float = 0.0

    @property
    def shares(self) -> int:
        return sum(l.shares for l in self.lots)

    def available(self, on: _date) -> int:
        """T+1：当天买入的不可卖。"""
        on = as_date(on)
        return sum(l.shares for l in self.lots if l.buy_date < on)

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price

    @property
    def avg_cost(self) -> float:
        return self.cost_amount / self.shares if self.shares else 0.0

    def holding_days(self, on: _date) -> float:
        """按股数加权的平均持有天数，用于红利税。"""
        total = self.shares
        if total == 0:
            return 0.0
        on = as_date(on)
        acc = sum(l.shares * (on - l.buy_date).days for l in self.lots)
        return acc / total

    def add(self, shares: int, on: date, cost: float) -> None:
        for lot in self.lots:
            if lot.buy_date == on:
                lot.shares += shares
                break
        else:
            self.lots.append(Lot(shares, on))
        self.cost_amount += cost

    def reduce(self, shares: int, on: date) -> None:
        """先进先出卖出。"""
        remaining = shares
        self.lots.sort(key=lambda l: l.buy_date)
        for lot in self.lots:
            if remaining <= 0:
                break
            take = min(lot.shares, remaining)
            lot.shares -= take
            remaining -= take
        self.lots = [l for l in self.lots if l.shares > 0]

        if self.shares == 0:
            self.cost_amount = 0.0
        else:
            # 成本按比例冲减
            ratio = shares / (self.shares + shares)
            self.cost_amount *= (1 - ratio)

    def split(self, ratio: float) -> None:
        """送股/转增/拆股：股数乘 ratio，总成本不变。"""
        for lot in self.lots:
            lot.shares = int(round(lot.shares * ratio))


# ---------------------------------------------------------------- 账户


class Account:
    def __init__(self, cash: float):
        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []

    # -- 估值
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def position(self, code: str) -> Position:
        if code not in self.positions:
            self.positions[code] = Position(code)
        return self.positions[code]

    def mark(self, prices: dict[str, float]) -> None:
        for code, px in prices.items():
            if code in self.positions:
                self.positions[code].last_price = px

    # -- 公司行为
    def apply_dividend(self, code: str, cash_per_share: float, on: _date) -> dict:
        """现金分红。按持股期限代扣个人所得税（差别化）。"""
        pos = self.positions.get(code)
        if pos is None or pos.shares == 0:
            return {"gross": 0.0, "tax": 0.0, "net": 0.0, "rate": 0.0}

        days = pos.holding_days(on)
        if days <= 30:
            rate = 0.20
        elif days <= 365:
            rate = 0.10
        else:
            rate = 0.0

        gross = pos.shares * cash_per_share
        tax = gross * rate
        net = gross - tax
        self.cash += net
        return {"gross": gross, "tax": tax, "net": net, "rate": rate, "hold_days": days}

    def apply_split(self, code: str, ratio: float) -> None:
        pos = self.positions.get(code)
        if pos:
            pos.split(ratio)


# ---------------------------------------------------------------- 撮合


class Broker:
    """按 A 股规则撮合委托。"""

    def __init__(self, account: Account, fees: FeeModel = DEFAULT_FEES,
                 volume_cap: float = 0.10):
        self.account = account
        self.fees = fees
        self.volume_cap = volume_cap
        self.rejections: list[Rejection] = []

    def _reject(self, order: Order, reason: str) -> Rejection:
        r = Rejection(order.code, order.side, order.qty, reason)
        self.rejections.append(r)
        return r

    def execute(self, order: Order, bar: Bar) -> Fill | Rejection:
        acc = self.account

        # 1) 停牌
        if bar.suspended:
            return self._reject(order, "suspended")

        # 2) 涨跌停：涨停买不到，跌停卖不出
        eps = 1e-6
        if order.side == "buy" and order.price >= bar.limit_up - eps:
            return self._reject(order, "limit_up")
        if order.side == "sell" and order.price <= bar.limit_down + eps:
            return self._reject(order, "limit_down")

        pos = acc.position(order.code)

        if order.side == "buy":
            qty = self._apply_lot_rule(order.qty)
            if qty <= 0:
                return self._reject(order, "below_one_lot")

            # 成交量占比上限（容量约束）
            max_qty = int(bar.volume * self.volume_cap) if math.isfinite(bar.volume) else qty
            max_qty = self._apply_lot_rule(max_qty)
            if max_qty <= 0:
                return self._reject(order, "volume_cap")
            qty = min(qty, max_qty)

            # 现金约束，不够就按手数缩量
            while qty > 0:
                amount = qty * order.price
                fee = self.fees.fee(amount, "buy")
                if amount + fee <= acc.cash + 1e-9:
                    break
                qty -= LOT_SIZE
            if qty <= 0:
                return self._reject(order, "insufficient_cash")

            amount = qty * order.price
            fee = self.fees.fee(amount, "buy")
            acc.cash -= amount + fee
            pos.add(qty, bar.date, amount + fee)
            pos.last_price = bar.close
            fill = Fill(order.code, "buy", qty, order.price, amount, fee, bar.date)
            acc.fills.append(fill)
            return fill

        # sell
        avail = pos.available(bar.date)
        if avail <= 0:
            return self._reject(order, "no_available_shares_t1")

        qty = min(order.qty, avail)
        # 零股只能一次性全部卖出
        if qty % LOT_SIZE != 0 and qty != avail:
            qty = (qty // LOT_SIZE) * LOT_SIZE
        if qty <= 0:
            return self._reject(order, "below_one_lot")

        amount = qty * order.price
        fee = self.fees.fee(amount, "sell")
        acc.cash += amount - fee
        pos.reduce(qty, bar.date)
        pos.last_price = bar.close
        fill = Fill(order.code, "sell", qty, order.price, amount, fee, bar.date)
        acc.fills.append(fill)
        return fill

    @staticmethod
    def _apply_lot_rule(qty: int) -> int:
        """买入必须是 100 股整数倍，向下取整。"""
        return int(qty // LOT_SIZE) * LOT_SIZE


# ---------------------------------------------------------------- 涨跌停价


def limit_prices(pre_close: float, limit_pct: float) -> tuple[float, float]:
    """按昨收和涨跌幅算涨跌停价（交易所四舍五入到分）。"""
    up = round(pre_close * (1 + limit_pct / 100), 2)
    down = round(pre_close * (1 - limit_pct / 100), 2)
    return up, down


def limit_pct_of(code: str, name: str = "") -> float:
    if code.startswith(("688", "689")):
        return 20.0
    if code.startswith(("300", "301", "302")):
        return 20.0
    if code.startswith(("4", "8", "9")):
        return 30.0
    return 5.0 if "ST" in name.upper() else 10.0

