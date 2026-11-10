"""Execute the pipeline notebooks in order, in place, with a run log (results/notebooks_run.log).

    uv run python scripts/run_notebooks.py            # 02 -> 03 -> 04
    uv run python scripts/run_notebooks.py 03 04      # a subset, in the given order

Stops at the first failing notebook; the traceback is in the executed notebook and in the log.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = {"01": "01_load_visium", "02": "02_load_bulk", "03": "03_train", "04": "04_predict_ood", "05": "05_ici_validation"}
TIMEOUT_SECONDS = 7200


def run(number: str) -> bool:
    path = ROOT / "notebooks" / f"{NOTEBOOKS[number]}.ipynb"
    command = [
        "uv", "run", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={TIMEOUT_SECONDS}", str(path),
    ]
    start = time.time()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    minutes = (time.time() - start) / 60
    if result.returncode != 0:
        logging.error("FAIL %s after %.1f min\n%s", path.name, minutes, result.stderr[-4000:])
        return False
    logging.info("OK %s in %.1f min", path.name, minutes)
    return True


def main(numbers: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(ROOT / "results" / "notebooks_run.log"), logging.StreamHandler(sys.stdout)],
    )
    for i, number in enumerate(numbers, 1):
        logging.info("[%d/%d] START %s", i, len(numbers), NOTEBOOKS[number])
        if not run(number):
            return 1
    logging.info("SUMMARY done=%d", len(numbers))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["02", "03", "04"]))
