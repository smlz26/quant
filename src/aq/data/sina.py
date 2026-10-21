"""新浪财经数据源：全 A 快照。

优势：一次分页就能拿到「价格、涨跌幅、成交额、换手率、总市值、流通市值、PE、PB」，
是全市场横截面最省事的免费源。
编码：GBK。
"""

from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Referer": "https://finance.sina.com.cn",
}

BASE = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
COUNT = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeStockCount?node=hs_a"
)


def _get(url: str, timeout: int = 20, retries: int = 4) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.2 + i)
    raise RuntimeError(f"新浪请求失败: {url[:120]}") from last


def stock_count() -> int:
    raw = _get(COUNT).decode("gbk", "ignore").strip().strip('"')
    try:
        return int(raw)
    except ValueError:
        return 0


def fetch_snapshot(num: int = 100, sleep: float = 0.18) -> pd.DataFrame:
    total = stock_count()
    rows: list[dict] = []
    page = 1
    while True:
        url = f"{BASE}?page={page}&num={num}&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=page"
        txt = _get(url).decode("gbk", "replace").strip()
        if txt in ("", "null", "[]"):
            break
        arr = json.loads(txt)
        if not arr:
            break
        rows.extend(arr)
        if total and len(rows) >= total:
            break
        page += 1
        time.sleep(sleep)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.rename(
        columns={
            "code": "code", "name": "name", "trade": "price",
            "settlement": "pre_close", "open": "open", "high": "high", "low": "low",
            "volume": "volume", "amount": "amount",
            "changepercent": "pct", "pricechange": "chg",
            "turnoverratio": "turnover", "per": "pe", "pb": "pb",
            "mktcap": "mkt_cap", "nmc": "float_cap",
        }
    )

    for col in ["price", "pre_close", "open", "high", "low", "volume", "amount",
                "pct", "chg", "turnover", "pe", "pb", "mkt_cap", "float_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["mkt_cap"] = df["mkt_cap"] * 1e4      # 万元 -> 元
    df["float_cap"] = df["float_cap"] * 1e4  # 万元 -> 元
    df["source"] = "sina"
    return df.reset_index(drop=True)


def fetch_quotes(symbols: list[str]) -> pd.DataFrame:
    """批量实时报价。symbols 形如 ['sh600276', 'sz000001']。"""
    rows = []
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        raw = _get("https://hq.sinajs.cn/list=" + ",".join(chunk)).decode("gbk", "replace")
        for line in raw.splitlines():
            if '="' not in line:
                continue
            head, body = line.split('="', 1)
            sym = head.split("_")[-1]
            parts = body.rstrip('";').split(",")
            if len(parts) < 32:
                continue
            rows.append({
                "symbol": sym, "name": parts[0], "open": parts[1],
                "pre_close": parts[2], "price": parts[3], "date": parts[30],
                "time": parts[31],
            })
    return pd.DataFrame(rows)
