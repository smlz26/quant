"""腾讯行情数据源：日线历史 K 线。

设计要点：
    1. 默认返回**不复权**价格 —— 回测引擎自己处理分红送转，不用复权价
    2. 双接口互备：kline/kline（不复权）优先，fqkline/get（可复权）兜底
    3. 磁盘缓存：网络挂了还能用今天早上抓的数据，不至于整个系统停摆

为什么必须有不复权价：
    涨跌停是按真实报价算的。用前复权价判断涨跌停，历史价格会被未来的
    分红改写，回测结论直接作废。
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Referer": "https://gu.qq.com/"}

FQ_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "meta" / "kline"


def _get(url: str, timeout: int = 20, retries: int = 2) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.8 + i)
    raise RuntimeError(f"腾讯请求失败: {url[:110]}") from last


def _parse(node: dict) -> list[list]:
    for key in ("day", "qfqday", "hfqday", "week", "qfqweek"):
        if key in node and node[key]:
            return node[key]
    return []


def _to_frame(rows: list[list]) -> pd.DataFrame:
    records = []
    for r in rows:
        if len(r) < 6:
            continue
        try:
            records.append({
                "date": r[0],
                "open": float(r[1]),
                "close": float(r[2]),
                "high": float(r[3]),
                "low": float(r[4]),
                "volume": float(r[5]),
            })
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["pct"] = df["close"].pct_change() * 100
    return df


def _cache_path(symbol: str, fq: str) -> Path:
    return CACHE_DIR / f"{symbol}_{fq or 'raw'}.csv"


def fetch_kline(symbol: str, count: int = 800, fq: str = "",
                max_age_hours: float = 12.0) -> pd.DataFrame:
    """拉日线。fq='' 不复权（默认）、'qfq' 前复权、'hfq' 后复权。

    缓存铁律：**只允许变长，不允许变短**。
    否则一个 count=5 的调用会把 800 根的缓存覆盖掉，
    后续基于长历史算出来的分位数、均线全部错位（这个坑踩过一次）。
    """
    path = _cache_path(symbol, fq)

    cached = None
    old_len = 0
    if path.exists():
        try:
            cached = pd.read_csv(path, parse_dates=["date"])
            old_len = len(cached)
        except Exception:  # noqa: BLE001
            cached = None
        if cached is not None:
            age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
            # 行数够 + 没过期，才能直接用
            if old_len >= count and age < timedelta(hours=max_age_hours):
                return cached

    urls = []
    if fq:
        urls = [f"{FQ_URL}?param={symbol},day,,,{count},{fq}",
                f"{KLINE_URL}?param={symbol},day,,,{count}"]
    else:
        urls = [f"{KLINE_URL}?param={symbol},day,,,{count}",
                f"{FQ_URL}?param={symbol},day,,,{count},qfq"]

    df = pd.DataFrame()
    for url in urls:
        try:
            payload = json.loads(_get(url).decode("utf-8", "replace"))
            node = ((payload or {}).get("data") or {}).get(symbol) or {}
            df = _to_frame(_parse(node))
            if not df.empty:
                break
        except Exception:  # noqa: BLE001
            continue

    if df.empty:
        return cached if cached is not None else df

    # 结果比缓存短（说明这次请求的 count 小），不要覆盖缓存
    if len(df) < old_len:
        return cached

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
