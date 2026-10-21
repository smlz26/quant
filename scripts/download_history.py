"""下载历史日线（新浪），落地到 data/raw/daily/{code}.csv。

可断点续传：已存在的文件直接跳过。
宇宙：按今日流通市值取前 N 只（沪深），这是 v1 的近似，幸存者偏差后面再修。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshots" / "snapshot_20261021.csv"
OUT = ROOT / "data" / "raw" / "daily"
OUT.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Referer": "https://finance.sina.com.cn"}
URL = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
       "CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={n}")

N_STOCKS = 700
DATALEN = 900


def to_symbol(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def fetch(symbol: str, n: int = DATALEN, retries: int = 3) -> pd.DataFrame:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(URL.format(sym=symbol, n=n), headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read().decode("utf-8", "replace").strip()
            if raw in ("", "null", "[]"):
                return pd.DataFrame()
            rows = json.loads(raw)
            df = pd.DataFrame(rows)
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.rename(columns={"day": "date"})
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 + i)
    raise RuntimeError(str(last)[:80])


def main() -> None:
    snap = pd.read_csv(SNAP, dtype={"code": str})
    snap["float_cap"] = pd.to_numeric(snap["float_cap"], errors="coerce")
    snap = snap[snap["code"].str.startswith(("0", "3", "6"))]
    snap = snap[snap["float_cap"].notna()].sort_values("float_cap", ascending=False).head(N_STOCKS)
    codes = snap["code"].tolist()
    print(f"目标 {len(codes)} 只", flush=True)

    todo = [c for c in codes if not ((OUT / f"{c}.csv").exists()
                                      and (OUT / f"{c}.csv").stat().st_size > 200)]
    skip = len(codes) - len(todo)
    print(f"待下载 {len(todo)}，已缓存 {skip}", flush=True)

    ok = fail = done = 0
    t0 = time.time()

    def job(code):
        try:
            df = fetch(to_symbol(code))
            if df.empty or len(df) < 250:
                return code, None
            return code, df
        except Exception:  # noqa: BLE001
            return code, None

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(job, c): c for c in todo}
        for fut in as_completed(futs):
            code, df = fut.result()
            if df is None:
                fail += 1
            else:
                df.to_csv(OUT / f"{code}.csv", index=False)
                ok += 1
            done += 1
            if done % 50 == 0:
                el = time.time() - t0
                rate = done / el if el else 0
                print(f"  {done}/{len(todo)}  ok={ok} fail={fail}  "
                      f"{el:.0f}s  {rate:.1f}/s", flush=True)

    print(f"完成 ok={ok} skip={skip} fail={fail} 用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

