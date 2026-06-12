"""``ndiff-web`` console entry point: serve the API + SPA and open a browser."""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path

from ndiff.server.app import STATIC_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ndiff-web", description="Launch the neutron-diffuse web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--data-root", default=None,
        help="Directory containing raw/ and processed/ "
             "(default: $NDIFF_DATA_ROOT or ./data)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (development)")
    args = parser.parse_args()

    if args.data_root:
        os.environ["NDIFF_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())

    if not STATIC_DIR.is_dir():
        print("[warn] SPA assets not found; only the API will be served. Build the "
              "frontend with:  (cd web && npm install && npm run build)", flush=True)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    print(f"neutron-diffuse web UI → {url}  (Ctrl-C to stop)", flush=True)
    uvicorn.run("ndiff.server.app:create_app", host=args.host, port=args.port,
                factory=True, reload=args.reload)


if __name__ == "__main__":
    main()
