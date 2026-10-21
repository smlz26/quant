"""云端每日任务入口（GitHub Actions 调用）。

流程：
    1. 判断今天是不是交易日（用上证指数的最后一根 K 线日期）
    2. 不是交易日 → 直接退出，不提交，避免产生无意义的提交
    3. 是交易日 → 重新生成 webapp/data.json

设计原则：**失败要静默降级**。抓不到数据就退出，绝不写入半成品，
让上一份正常的 data.json 继续对外服务。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def is_trading_day() -> tuple[bool, str]:
    from aq.data import tencent
    try:
        k = tencent.fetch_kline("sh000001", count=5, max_age_hours=0)
    except Exception as exc:  # noqa: BLE001
        return False, f"取上证指数失败: {exc}"
    if k.empty:
        return False, "上证指数 K 线为空"
    last = k["date"].iloc[-1].date()
    return last == date.today(), f"指数最后交易日 {last}，今天 {date.today()}"


def main() -> None:
    ok, msg = is_trading_day()
    print(f"[交易日判断] {msg}")

    if not ok:
        print("非交易日或行情尚未更新，跳过本次更新。")
        return

    import build_app_data
    print("[生成] 开始重建 data.json ...")
    build_app_data.main()

    out = ROOT / "webapp" / "data.json"
    if not out.exists() or out.stat().st_size < 1000:
        print("生成结果异常，退出。", file=sys.stderr)
        sys.exit(1)
    print(f"[完成] {out}  {out.stat().st_size} 字节")


if __name__ == "__main__":
    main()
