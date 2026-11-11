import numpy as np
import pandas as pd

from tls_model import io_geomx, prepare


def _rois():
    counts = pd.DataFrame({"R1": [10, 0, 5], "R2": [1, 2, 3], "R3": [4, 4, 4]},
                          index=pd.Index(["A2M", "CD19", "MS4A1"], name="GeneFeature"))
    metadata = pd.DataFrame({"Sample_name": ["R1", "R2", "R3"], "Patient": ["P1", "P1", "P2"],
                             "Sample_type": ["TLS", "control", "TLS"]})
    adata = io_geomx.gse271255_units(counts, metadata)
    adata.obs.loc["GSE271255:roi:R3", "excluded"] = True
    return adata


def test_geomx_units_follow_the_unit_schema_and_drop_excluded_rois():
    genes = ["CD19", "MS4A1", "CXCL13"]

    units, coverage = prepare.geomx_units(_rois(), genes, split="ood")

    assert list(units.obs.columns) == [c for c in prepare.UNIT_COLUMNS if c not in ("role", "tls_three")]
    assert list(units.obs_names) == ["GSE271255:roi:R1", "GSE271255:roi:R2"]
    assert set(units.obs["kind"]) == {"geomx"} and set(units.obs["unit"]) == {"roi"}
    assert units.obs["patient"].tolist() == ["GSE271255:P1", "GSE271255:P1"]
    assert units.obs["cohort"].tolist() == ["GSE271255"] * 2 and units.obs["group"].tolist() == ["GSE271255"] * 2
    assert units.obs["split"].tolist() == ["ood"] * 2 and units.obs["platform"].tolist() == ["geomx"] * 2
    assert units.obs["tls_any"].tolist() == [1.0, 0.0] and units.obs["tls_mature"].isna().all()
    assert set(units.obs["label_source"]) == {"pathologist"}
    assert list(units.var_names) == genes and (units.X[:, 2] == 0).all()
    assert np.isclose(coverage, 2 / 3)


def test_geomx_split_from_config():
    cfg = {"cohorts": {"geomx_train": ["GSE271255"], "geomx_holdout": ["GSE276935"]}}

    assert prepare.geomx_split(cfg, "GSE271255") == "train"
    assert prepare.geomx_split(cfg, "GSE276935") == "ood"
