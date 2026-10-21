"""交易所官方数据源：股票主档（最权威）。

为什么必须接入交易所官网而不是只用行情站：
    · 上市日期 —— 决定「次新股过滤」，行情站的这个字段经常是错的
    · 退市日期 —— 决定「幸存者偏差」是否被消除
    · 总股本 / 流通股本 —— 决定市值类因子，且能和行情站的市值做交叉校验
    · 板块归属 —— 决定涨跌停幅度（主板 10% / 创业板 20% / 北交所 30%）

深交所：CATALOGID=1110（A 股列表）
上交所：sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L
"""

from __future__ import annotations

import json
import re
import time
import urllib.request

import pandas as pd

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SZSE_URL = "http://www.szse.cn/api/report/ShowReport/data"
SSE_URL = "http://query.sse.com.cn/commonQuery.do"


def _get(url: str, referer: str, timeout: int = 20, retries: int = 3) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Referer": referer}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 + i)
    raise RuntimeError(f"交易所请求失败: {url[:110]}") from last


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", str(s)).strip()


def fetch_szse_list(max_pages: int = 200) -> pd.DataFrame:
    """深交所 A 股列表（含创业板）。"""
    rows: list[dict] = []
    page = 1
    page_count = 1
    while page <= min(page_count, max_pages):
        url = (
            f"{SZSE_URL}?SHOWTYPE=JSON&CATALOGID=1110&TABKEY=tab1"
            f"&PAGENO={page}&random=0.5"
        )
        payload = json.loads(_get(url, "http://www.szse.cn/").decode("utf-8", "replace"))
        block = payload[0]
        meta = block.get("metadata", {})
        page_count = int(meta.get("pagecount") or 1)
        batch = block.get("data") or []
        if not batch:
            break
        rows.extend(batch)
        page += 1
        time.sleep(0.12)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    out = pd.DataFrame({
        "code": df["agdm"].astype(str).str.zfill(6),
        "name": df["agjc"].map(_strip_html),
        "list_date": pd.to_datetime(df["agssrq"], errors="coerce"),
        "total_shares": pd.to_numeric(df["agzgb"], errors="coerce") * 1e8,
        "float_shares": pd.to_numeric(df["agltgb"], errors="coerce") * 1e8,
        "industry_csrc": df.get("sshymc", ""),
        "board": df["bk"],
        "exchange": "SZSE",
    })
    return out


def fetch_sse_list(page_size: int = 1000, max_pages: int = 30) -> pd.DataFrame:
    """上交所 A 股列表（含科创板）。注意 pageSize 有上限，必须翻页。"""
    rows: list[dict] = []
    total = None
    for page in range(1, max_pages + 1):
        url = (
            f"{SSE_URL}?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L&isPagination=true"
            f"&pageHelp.pageSize={page_size}&pageHelp.pageNo={page}"
            f"&pageHelp.beginPage={page}&pageHelp.cacheSize=1"
        )
        payload = json.loads(_get(url, "http://www.sse.com.cn/").decode("utf-8", "replace"))
        ph = payload.get("pageHelp") or {}
        batch = ph.get("data") or []
        total = int(ph.get("total") or total or 0)
        if not batch:
            break
        rows.extend(batch)
        if total and len(rows) >= total:
            break
        time.sleep(0.12)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    out = pd.DataFrame({
        "code": df["A_STOCK_CODE"].astype(str).str.zfill(6),
        "name": df["SEC_NAME_CN"],
        "list_date": pd.to_datetime(df["LIST_DATE"], format="%Y%m%d", errors="coerce"),
        "delist_date": pd.to_datetime(df["DELIST_DATE"], format="%Y%m%d", errors="coerce"),
        "industry_csrc": df.get("CSRC_CODE_DESC", ""),
        "stock_type": df.get("STOCK_TYPE", ""),
        "state_code": df.get("STATE_CODE", ""),
        "exchange": "SSE",
    })

    def board_of(row):
        code = row["code"]
        if code.startswith(("688", "689")):
            return "科创板"
        return "主板"

    out["board"] = out.apply(board_of, axis=1)
    return out


def fetch_all(max_pages: int = 200) -> pd.DataFrame:
    """沪深两市主板 + 创业板 + 科创板完整主档。"""
    sz = fetch_szse_list(max_pages=max_pages)
    sse = fetch_sse_list()

    for col in ("total_shares", "float_shares", "delist_date", "stock_type", "state_code"):
        if col not in sz.columns:
            sz[col] = pd.NA
        if col not in sse.columns:
            sse[col] = pd.NA

    common = ["code", "name", "list_date", "total_shares", "float_shares",
              "industry_csrc", "board", "exchange", "delist_date"]
    df = pd.concat([sz[common], sse[common]], ignore_index=True)
    return df.drop_duplicates(subset="code").reset_index(drop=True)
