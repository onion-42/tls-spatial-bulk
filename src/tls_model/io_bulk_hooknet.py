"""HookNet-TLS manual annotations (Zenodo 10614928; van Rijthoven et al.) as bulk TLS labels, and the UCSC Xena
expression matrices of the two new TCGA cohorts in the archive layout.

Annotations: one ASAP XML per TCGA diagnostic slide, polygons grouped as `tls` / `gc` (germinal centre) / `lymph
node` / `other` / ...; `annotations.json` lists the slides per cohort (LUSC / KIRC / BLCA). Labels are per sample
(slide id[:15], e.g. TCGA-34-8455-01); a patient with two slides takes the maximum count over them. TLS+ = at least
one TLS polygon; the strict label = at least one GC polygon (GC-anchored TLS); maturity: GC = mature, TLS without GC
= immature, no TLS = no maturity.

Caveat (Silina lab): every TCGA label, HookNet included, is read on the diagnostic FFPE slide while the RNA comes
from a different frozen piece of the tumour. These labels are holdout / test labels only and are never fitted.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from tls_model.labels import IMMATURE, MATURE

log = logging.getLogger(__name__)

METHOD = "HookNet-TLS_manual"
PRIORITY = 2  # after the archive pathologists (priority 1), before the HookNet CNN runs (3, 4)
XML_DIR = "annotations_approx_05um"
INDEX_FILE = "annotations.json"
COHORTS = {"BLCA": "TCGA-BLCA", "LUSC": "TCGA-LUSC", "KIRC": "TCGA-KIRC/KIRP/KICH"}
SAMPLE_CHARS, PATIENT_CHARS = 15, 12
PRIMARY_TUMOUR = "01"
STATUS = {True: "positive", False: "negative"}


def count_groups(xml_path: Path) -> dict[str, int]:
    """Number of polygons per annotation group in one ASAP XML."""
    counts: dict[str, int] = {}
    for node in ET.parse(xml_path).getroot().iter("Annotation"):
        group = node.get("PartOfGroup", "")
        counts[group] = counts.get(group, 0) + 1
    return counts


def _slide_rows(zenodo_dir: Path) -> pd.DataFrame:
    index = json.loads((zenodo_dir / INDEX_FILE).read_text(encoding="utf-8"))
    rows = []
    for cohort, files in index.items():
        for name in files:
            counts = count_groups(zenodo_dir / XML_DIR / name)
            slide = Path(name).stem
            rows.append({"cancer_type": cohort, "cohort": COHORTS[cohort], "slide": slide,
                         "sample_id": slide[:SAMPLE_CHARS], "patient_id": slide[:PATIENT_CHARS],
                         "n_tls": counts.get("tls", 0), "n_gc": counts.get("gc", 0)})
    return pd.DataFrame(rows)


def hooknet_labels(zenodo_dir: Path) -> pd.DataFrame:
    """One label row per TCGA sample in the archive `labels_primary` layout (plus TLS_status_strict, n_slides)."""
    slides = _slide_rows(Path(zenodo_dir))
    per_sample = (slides.groupby(["cancer_type", "cohort", "sample_id", "patient_id"], sort=True)
                  .agg(TLS_count=("n_tls", "max"), TLS_mature_count=("n_gc", "max"), n_slides=("slide", "size"))
                  .reset_index())
    has_tls, has_gc = per_sample["TLS_count"] >= 1, per_sample["TLS_mature_count"] >= 1
    maturity = pd.Series("unknown", index=per_sample.index)
    maturity[has_tls] = IMMATURE
    maturity[has_gc] = MATURE
    table = per_sample.assign(
        subtype="", method=METHOD, priority=PRIORITY, n_annotators=1,
        TLS_status=has_tls.map(STATUS), TLS_status_strict=has_gc.map(STATUS), maturity_status=maturity,
        has_maturity=has_tls, expression_id=per_sample["sample_id"], has_expression=False, excluded=False,
    )
    log.info("HookNet labels: %s", table.groupby("cohort")[["TLS_status", "TLS_status_strict"]]
             .agg(lambda s: int((s == "positive").sum())).to_dict("index"))
    return table


def xena_to_archive(xena: pd.DataFrame, probemap: pd.DataFrame, archive_genes: list[str]) -> pd.DataFrame:
    """Xena GDC matrix (Ensembl ids × samples, log2(TPM + 1)) -> archive layout (samples × HUGO genes).

    Primary-tumour vials only (`-01A`, `-01B`, ...), renamed to the 15-character sample id, first vial per sample;
    symbols mapped through the probemap, duplicate symbols = max; genes missing from Xena = 0 (not detected)."""
    symbols = probemap.drop_duplicates("id").set_index("id")["gene"]
    # GENCODE lists the pseudo-autosomal copies as <id>.<version>_PAR_Y: one symbol per version-free id is enough.
    version_free = pd.Series(symbols.to_numpy(), index=symbols.index.str.split(".").str[0])
    version_free = version_free[~version_free.index.duplicated()]
    ids = pd.Index(xena.index.astype(str))
    gene = ids.map(symbols).to_series(index=ids)
    unmapped = gene.isna()
    gene[unmapped] = ids[unmapped].str.split(".").str[0].map(version_free)
    matrix = xena[gene.notna().to_numpy()].set_axis(gene.dropna().to_numpy(), axis=0)
    matrix = matrix.groupby(level=0).max()
    is_primary = matrix.columns.str[13:15] == PRIMARY_TUMOUR
    matrix = matrix.loc[:, is_primary].set_axis(matrix.columns[is_primary].str[:SAMPLE_CHARS], axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    out = matrix.reindex(index=archive_genes, fill_value=0.0).T.astype(float)
    out.index.name = "expression_id"
    log.info("xena: %d samples, %d / %d archive genes present", len(out), int(matrix.index.isin(archive_genes).sum()),
             len(archive_genes))
    return out
