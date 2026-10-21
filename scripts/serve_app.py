"""启动手机端看板服务。

用法：双击 start.bat，或命令行运行本文件。
然后用手机浏览器打开打印出来的局域网地址。
"""

from __future__ import annotations

import http.server
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
PORT = 8765


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("223.5.5.5", 80))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEBAPP), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, *args):
        pass


def main() -> None:
    if not (WEBAPP / "data.json").exists():
        print("缺少 data.json，先运行：python scripts/build_app_data.py")
        sys.exit(1)

    ip = lan_ip()
    print("=" * 58)
    print("  量化看板 已启动")
    print("=" * 58)
    print(f"  手机访问（同一 WiFi）:  http://{ip}:{PORT}")
    print(f"  本机访问            :  http://127.0.0.1:{PORT}")
    print()
    print("  手机浏览器打开后 →「添加到主屏幕」即可像 App 一样用")
    print("  Ctrl+C 停止")
    print("=" * 58)

    with http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")


if __name__ == "__main__":
    main()
