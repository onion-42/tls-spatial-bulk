import numpy as np
import pandas as pd

from tls_model import io_geomx


def _gse271255():
    counts = pd.DataFrame({"A_GBM_002_TLS": [10, 0, 5], "A_GBM_003_control": [1, 2, 3], "B_GBM_001_TLS": [4, 4, 4]},
                          index=pd.Index(["A2M", "CD19", "MS4A1"], name="GeneFeature"))
    metadata = pd.DataFrame({"Sample_name": ["A_GBM_002_TLS", "A_GBM_003_control", "B_GBM_001_TLS"],
                             "Patient": ["PAT-511", "PAT-511", "PAT-409"], "Sample_type": ["TLS", "control", "TLS"]})
    return counts, metadata


def test_gse271255_units_label_tls_vs_control():
    counts, metadata = _gse271255()

    adata = io_geomx.gse271255_units(counts, metadata)

    assert list(adata.obs_names) == ["GSE271255:roi:A_GBM_002_TLS", "GSE271255:roi:A_GBM_003_control",
                                     "GSE271255:roi:B_GBM_001_TLS"]
    assert adata.obs["tls"].tolist() == [1.0, 0.0, 1.0]
    assert adata.obs["tls_mature"].isna().all()
    assert adata.obs["patient"].tolist() == ["PAT-511", "PAT-511", "PAT-409"]
    assert adata.obs["group"].tolist() == ["GSE271255"] * 3
    assert (adata.obs["source"] == "GSE271255").all() and not adata.obs["excluded"].any()
    assert adata.obs["annotation"].tolist() == ["TLS", "control", "TLS"]
    assert list(adata.var_names) == ["A2M", "CD19", "MS4A1"] and adata.X[0].tolist() == [10, 0, 5]


def _gse276935():
    names = ["41Ovary001CD20", "41Ovary001Immune", "58AOmentum002Stroma", "58BFallopian Tube003Tumor",
             "58COvary004Full ROI", "41Ovary005Other", "LNLymph Node006CD20"]
    counts = pd.DataFrame(np.arange(21, dtype=float).reshape(3, 7), index=pd.Index(["ITK", "CD19", "MS4A1"], name="TargetName"),
                          columns=names)
    segments = pd.DataFrame({"TargetName": names, "ROILabel": [1, 1, 2, 3, 4, 5, 6],
                             "SegmentLabel": ["CD20", "Immune", "Stroma", "Tumor", "Full ROI", "Other", "CD20"],
                             "QCFlags": ["", "Low Nuclei Count", "", "", "", "", ""]})
    return counts, segments


def test_gse276935_segment_mapping_patient_and_lymph_node_exclusion():
    counts, segments = _gse276935()

    adata = io_geomx.gse276935_units(counts, segments)
    obs = adata.obs.set_index("roi")

    assert obs["tls"].tolist()[:4] == [1.0, 0.0, 0.0, 0.0]
    assert np.isnan(obs.loc["58COvary004Full ROI", "tls"])  # Full ROI: unlabelled, predict only
    assert obs["excluded"].tolist() == [False, False, False, False, False, True, True]  # Other, lymph node slide
    assert obs["group"].tolist()[:5] == ["41Ovary", "41Ovary", "58AOmentum", "58BFallopian Tube", "58COvary"]
    assert obs["patient"].tolist()[:5] == ["41", "41", "58", "58", "58"]  # one patient, three sites
    assert obs["qc_flagged"].tolist() == [False, True, False, False, False, False, False]
    assert obs["annotation"].tolist()[:2] == ["CD20", "Immune"]
    assert list(adata.obs_names)[:1] == ["GSE276935:roi:41Ovary001CD20"]
    assert adata.X[0].tolist() == [0.0, 7.0, 14.0]  # ROIs x genes


def test_roi_log2cpm_missing_model_genes_are_zero_and_coverage_reported():
    counts, metadata = _gse271255()
    adata = io_geomx.gse271255_units(counts, metadata)

    values, coverage = io_geomx.roi_log2cpm(adata, ["CD19", "MS4A1", "CXCL13", "CCL19"])

    assert values.shape == (3, 4)
    assert coverage == 0.5
    assert (values[:, 2:] == 0).all()
    assert values[0, 0] == 0.0 and values[0, 1] > 0  # CD19 = 0 counts, MS4A1 = 5 of 15 total counts
    assert np.isclose(values[0, 1], np.log2(5 / 15 * 1e6 + 1))
