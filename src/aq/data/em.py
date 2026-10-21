"""东方财富行情接口封装 —— 零第三方依赖（仅标准库 + pandas）。

为什么要自己写而不用 akshare：
  akshare 是爬虫的二次封装，上游页面一改就坏，且无法控制重试/限速。
  直接对接东财 JSON 接口，稳定性和可控性都更好。
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import pandas as pd

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}

# 全 A：深主板 + 创业板 + 沪主板 + 科创板 + 北交所
FS_ALL_A = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"

SNAPSHOT_FIELDS = (
    "f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,"
    "f20,f21,f23,f62,f100,f184"
)

COLUMN_MAP = {
    "f2": "price", "f3": "pct", "f4": "chg", "f5": "volume", "f6": "amount",
    "f7": "amplitude", "f8": "turnover", "f9": "pe", "f10": "vol_ratio",
    "f12": "code", "f13": "market", "f14": "name", "f15": "high", "f16": "low",
    "f17": "open", "f18": "pre_close", "f20": "mkt_cap", "f21": "float_cap",
    "f23": "pb", "f62": "main_net_inflow", "f100": "industry", "f184": "main_net_pct",
}

INDEX_SECIDS = {
    "上证指数": "1.000001",
    "深证成指": "0.399001",
    "沪深300": "1.000300",
    "中证500": "1.000905",
    "中证1000": "1.000852",
    "中证2000": "0.932000",
    "创业板指": "0.399006",
    "科创50": "1.000688",
}


def _get(url: str, timeout: int = 20, retries: int = 3) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - 网络异常统一重试
            last = exc
            time.sleep(1.0 + i)
    raise RuntimeError(f"请求失败: {url[:120]}") from last


def _json(url: str):
    return json.loads(_get(url))


def _num(v):
    """东财用 '-' 表示缺失。"""
    if v in ("-", "", None):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fetch_a_snapshot(page_size: int = 200, sleep: float = 0.15) -> pd.DataFrame:
    """拉取全 A 实时快照（收盘后即为当日收盘数据）。"""
    rows: list[dict] = []
    page = 1
    total = None
    while True:
        # 注意：fs 里的 "+" 是多市场分隔符，必须保持原样，不能被 urlencode 成 %2B
        query = (
            f"pn={page}&pz={page_size}&po=1&np=1&fltt=2&invt=2&fid=f12"
            f"&fs={urllib.parse.quote(FS_ALL_A, safe=':+,.')}"
            f"&fields={SNAPSHOT_FIELDS}"
        )
        payload = _json("https://push2.eastmoney.com/api/qt/clist/get?" + query)
        data = (payload or {}).get("data") or {}
        diff = data.get("diff") or []
        if isinstance(diff, dict):
            diff = list(diff.values())
        if not diff:
            break
        rows.extend(diff)
        total = data.get("total") or total
        if total and len(rows) >= int(total):
            break
        page += 1
        time.sleep(sleep)

    df = pd.DataFrame(rows).rename(columns=COLUMN_MAP)
    for col in df.columns:
        if col not in ("code", "name", "industry"):
            df[col] = df[col].map(_num)
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def fetch_kline(secid: str, beg: str = "20180101", end: str = "20991231",
                klt: int = 101, fqt: int = 0) -> pd.DataFrame:
    """日线 K 线。klt=101 日线；fqt=0 不复权 / 1 前复权 / 2 后复权。"""
    query = urllib.parse.urlencode(
        {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": klt, "fqt": fqt, "beg": beg, "end": end, "lmt": "1000000",
        }
    )
    payload = _json("https://push2his.eastmoney.com/api/qt/stock/kline/get?" + query)
    klines = ((payload or {}).get("data") or {}).get("klines") or []

    records = []
    for line in klines:
        parts = line.split(",")

        def num(i):
            try:
                return float(parts[i])
            except (IndexError, ValueError):
                return float("nan")

        records.append(
            {
                "date": parts[0],
                "open": num(1), "close": num(2), "high": num(3), "low": num(4),
                "volume": num(5), "amount": num(6), "amplitude": num(7),
                "pct": num(8), "chg": num(9), "turnover": num(10),
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def board_of(code: str) -> str:
    if code.startswith(("688", "689")):
        return "科创板"
    if code.startswith(("300", "301", "302")):
        return "创业板"
    if code.startswith(("8", "4", "9")):
        return "北交所"
    return "主板"


def limit_pct(code: str, name: str) -> float:
    """涨跌停幅度（百分数）。"""
    board = board_of(code)
    if board == "北交所":
        return 30.0
    if board in ("科创板", "创业板"):
        return 20.0
    # 主板 ST 为 5%，其余 10%
    return 5.0 if "ST" in name.upper() else 10.0
