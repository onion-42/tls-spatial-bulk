"""GeoMx supplementary files of the two ROI series from GEO into data/private/geomx (gitignored). Logged,
idempotent (existing files are kept).

    uv run python scripts/fetch_geomx.py
"""

from __future__ import annotations

import logging
import sys
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tls_model import io_geomx  # noqa: E402

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
SERIES_DIR = {"GSE271255": "GSE271nnn", "GSE276935": "GSE276nnn"}
OUT_DIR = ROOT / "data" / "private" / "geomx"
log = logging.getLogger(__name__)


def fetch(source: str, name: str) -> None:
    dest = OUT_DIR / name
    if dest.exists():
        log.info("SKIP %s (exists)", name)
        return
    url = f"{GEO}/{SERIES_DIR[source]}/{source}/suppl/{name}"
    log.info("GET %s", url)
    with urllib.request.urlopen(url, timeout=600) as response, dest.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 20):
            handle.write(chunk)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "results" / "fetch_geomx.log"), logging.StreamHandler(sys.stdout)])
    failed = []
    for i, (source, files) in enumerate(io_geomx.FILES.items(), 1):
        log.info("[%d/%d] START %s", i, len(io_geomx.FILES), source)
        try:
            for name in files:
                fetch(source, name)
            log.info("[%d/%d] OK %s", i, len(io_geomx.FILES), source)
        except Exception:  # noqa: BLE001 - one series must not stop the other
            failed.append(source)
            log.error("[%d/%d] FAIL %s\n%s", i, len(io_geomx.FILES), source, traceback.format_exc())
    log.info("SUMMARY done=%d failed=%d", len(io_geomx.FILES) - len(failed), len(failed))
    if failed:
        log.error("FAILED: %s", failed)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
