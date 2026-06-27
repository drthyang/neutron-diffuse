# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""``nebula3d-web`` console entry point: serve the API + SPA and open a browser."""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path

from nebula3d.server.app import STATIC_DIR

# Source extensions whose mtime determines whether the built SPA is stale.
_SPA_SRC_SUFFIXES = frozenset({".ts", ".tsx", ".css", ".html"})


def _warn_if_spa_stale() -> None:
    """In a source checkout, warn when the built SPA is older than ``web/src``.

    ``server/static`` is a gitignored build artifact (``npm run build``); it
    silently goes stale when the frontend changes without a rebuild, so the
    native server can serve an old UI (a real footgun).  This only fires when
    ``web/src`` exists — i.e. a dev checkout — so it is a no-op for installed
    wheels, and never raises (best-effort).
    """
    try:
        web_src = STATIC_DIR.parents[3] / "web" / "src"
        built = STATIC_DIR / "index.html"
        if not web_src.is_dir() or not built.exists():
            return
        built_mtime = built.stat().st_mtime
        newest_src = max(
            (p.stat().st_mtime for p in web_src.rglob("*")
             if p.suffix in _SPA_SRC_SUFFIXES),
            default=0.0,
        )
        if newest_src > built_mtime:
            print(
                "[warn] The web UI bundle looks out of date — web/src changed "
                "after the last build. Rebuild it with:  make ui   "
                "(or: cd web && npm run build), then hard-refresh the browser.",
                flush=True,
            )
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nebula3d-web", description="Launch the nebula3d web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--data-root", default=None,
        help="Directory containing raw/ and processed/ "
             "(default: $NEBULA3D_DATA_ROOT or ./data)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (development)")
    args = parser.parse_args()

    if args.data_root:
        os.environ["NEBULA3D_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())

    if not STATIC_DIR.is_dir():
        print("[warn] SPA assets not found; only the API will be served. Build the "
              "frontend with:  make ui   (or: cd web && npm install && npm run build)",
              flush=True)
    else:
        _warn_if_spa_stale()

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    print(f"nebula3d web UI → {url}  (Ctrl-C to stop)", flush=True)
    uvicorn.run("nebula3d.server.app:create_app", host=args.host, port=args.port,
                factory=True, reload=args.reload)


if __name__ == "__main__":
    main()
