import gzip

import numpy as np
import pandas as pd

from tls_model import io_visium


def test_load_gse_bulk_sums_duplicate_symbols_then_logs(tmp_path):
    folder = tmp_path / "gse175540"
    folder.mkdir()
    with gzip.open(folder / "bulk_TPM.csv.gz", "wt") as handle:
        handle.write(";a_1;b_7\nGENE1;1;3\nDUP;2;0\nDUP;5;1\n")

    out = io_visium.load_gse_bulk(tmp_path)

    assert list(out.index) == ["a_1", "b_7"]
    assert out.columns.is_unique and sorted(out.columns) == ["DUP", "GENE1"]
    assert np.isclose(out.loc["a_1", "DUP"], np.log2(7 + 1))
    assert np.isclose(out.loc["b_7", "GENE1"], np.log2(3 + 1))


def test_spot_obs_joins_positions_and_labels_by_barcode():
    barcodes = pd.Index(["B2", "B1", "B3"])
    positions = pd.DataFrame(
        {
            "barcode": ["B1", "B2", "B3", "B4"],
            "in_tissue": [1, 1, 1, 0],
            "array_row": [0, 1, 2, 3],
            "array_col": [0, 1, 2, 3],
            "pxl_row": [10, 11, 12, 13],
            "pxl_col": [20, 21, 22, 23],
        }
    )
    annotation = pd.DataFrame({"Barcode": ["B1", "B2"], "TLS_2_cat": ["TLS", "NO_TLS"]})

    obs, spatial = io_visium.spot_obs(barcodes, positions, annotation)

    assert obs.index.tolist() == ["B2", "B1", "B3"]
    assert obs["label_raw"].tolist() == ["NO_TLS", "TLS", ""]
    assert obs["tls"].tolist() == [0, 1, 0]
    assert obs["array_row"].tolist() == [1, 0, 2]
    assert spatial.tolist() == [[21, 11], [20, 10], [22, 12]]  # (x, y) = (pxl_col, pxl_row)


def test_gse_sections_skip_unannotated_and_map_gsm():
    sections = io_visium.gse_sections()

    names = [s.name for s in sections]
    assert len(sections) == 23
    assert "ffpe_c_10" not in names
    by_name = {s.name: s for s in sections}
    assert by_name["ffpe_c_2"].gsm == "GSM5924030"
    assert by_name["frozen_a_1"].gsm == "GSM5924042"
    assert by_name["frozen_c_57"].gsm == "GSM5924053"


def test_gse_file_urls_follow_geo_layout():
    section = {s.name: s for s in io_visium.gse_sections()}["frozen_a_1"]

    urls = io_visium.gse_file_urls(section)

    assert len(urls) == len(io_visium.GSE_FILES)
    assert urls["filtered_feature_bc_matrix.h5"] == (
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5924nnn/GSM5924042/suppl/"
        "GSM5924042_frozen_a_1_filtered_feature_bc_matrix.h5"
    )


def test_section_metadata_patient_and_protocol():
    by_name = {s.name: s for s in io_visium.gse_sections()}

    assert by_name["ffpe_c_2"].patient == by_name["frozen_c_2"].patient == "GSE175540_c_2"
    assert by_name["ffpe_c_2"].protocol == "ffpe"
    assert by_name["frozen_b_7"].protocol == "frozen"
    assert by_name["frozen_b_7"].bulk_id == "b_7"
    assert by_name["frozen_c_5"].bulk_id is None
