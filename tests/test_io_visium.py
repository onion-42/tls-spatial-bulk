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

    urls = io_visium.section_file_urls(section)

    assert len(urls) == len(io_visium.GEO_FILES)
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


def _write_positions(path, header: bool):
    import gzip as gz

    lines = ["B1,1,0,0,10,20", "B2,1,1,1,11,21"]
    if header:
        lines = ["barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres"] + lines
    with gz.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")


def test_read_positions_with_and_without_header_are_identical(tmp_path):
    _write_positions(tmp_path / "old.csv.gz", header=False)
    _write_positions(tmp_path / "new.csv.gz", header=True)

    old = io_visium.read_positions(tmp_path / "old.csv.gz")
    new = io_visium.read_positions(tmp_path / "new.csv.gz")

    assert list(old.columns) == list(new.columns) == io_visium.POSITION_COLUMNS
    pd.testing.assert_frame_equal(old, new)
    assert old["barcode"].tolist() == ["B1", "B2"] and old["pxl_col"].tolist() == [20, 21]


def test_new_cohorts_sections_counts_gsm_and_patients():
    rcc = {s.name: s for s in io_visium.cohort_sections("GSE273952")}
    breast = {s.name: s for s in io_visium.cohort_sections("GSE351843")}

    assert len(rcc) == 8 and len(breast) == 33
    assert not set(rcc) & set(io_visium.GSE273952_DUPLICATES)  # re-annotated GSE175540 sections are not a cohort
    assert rcc["P_10"].gsm == "GSM8440303" and rcc["P_19"].gsm == "GSM8440312"
    assert rcc["P_10"].patient == "GSE273952_P_10" and rcc["P_10"].section == "GSE273952_P_10"
    assert breast["P01_Vis"].gsm == "GSM10144987" and breast["P32_Vis"].gsm == "GSM10145019"
    assert breast["P29_Vis_Rep1"].gsm == "GSM10145015" and breast["P29_Vis_Rep2"].gsm == "GSM10145016"
    assert breast["P29_Vis_Rep1"].patient == breast["P29_Vis_Rep2"].patient == "GSE351843_P29"
    assert len({s.patient for s in breast.values()}) == 32
    assert all(s.bulk_id is None for s in list(rcc.values()) + list(breast.values()))
    assert {s.protocol for s in rcc.values()} == {s.protocol for s in breast.values()} == {"ffpe"}


def test_new_cohort_file_urls_follow_geo_layout():
    breast = {s.name: s for s in io_visium.cohort_sections("GSE351843")}

    urls = io_visium.section_file_urls(breast["P29_Vis_Rep1"])

    assert urls["spatial_tissue_positions.csv.gz"] == (
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM10145nnn/GSM10145015/suppl/"
        "GSM10145015_P29_Vis_Rep1_spatial_tissue_positions.csv.gz"
    )
    assert "TLS_annotation.csv.gz" not in urls  # breast labels come from Zenodo, not GEO


def test_breast_annotation_uses_immune_column(tmp_path):
    section = {s.name: s for s in io_visium.cohort_sections("GSE351843")}["P02_Vis"]
    folder = tmp_path / "gse351843" / "zenodo" / "pathology_annotations" / "visium"
    folder.mkdir(parents=True)
    (folder / "P02_Vis_pathology_annotations.csv").write_text(
        ",epithelial,immune,stroma,additional_features,single_layer\n"
        "B1,invasive,,desmoplastic,,invasive\nB2,,tls,,,tls\nB3,,aggregate,,,aggregate\n"
    )

    table = io_visium.annotation_table(section, tmp_path)

    assert table.shape == (3, 2)
    assert table.iloc[:, 0].tolist() == ["B1", "B2", "B3"]
    assert table.iloc[:, 1].tolist() == ["", "tls", "aggregate"]


def test_geo_annotation_uses_the_cohort_label_column(tmp_path):
    section = {s.name: s for s in io_visium.cohort_sections("GSE273952")}["P_16"]
    folder = tmp_path / "gse273952" / "P_16"
    folder.mkdir(parents=True)
    with gzip.open(folder / "TLS_annotation.csv.gz", "wt") as handle:
        handle.write("Barcode,TLS_annotation\nB1,NO_TLS\nB2,TLS_SFL\n")

    table = io_visium.annotation_table(section, tmp_path)

    assert table.iloc[:, 1].tolist() == ["NO_TLS", "TLS_SFL"]


def test_download_jobs_cover_all_cohorts_and_archives():
    jobs = io_visium.download_jobs(__import__("pathlib").Path("raw"))

    dests = {str(dest) for _, dest in jobs}
    assert "raw/gse175540/ffpe_c_2/tissue_positions_list.csv.gz" in dests
    assert "raw/gse273952/P_10/TLS_annotation.csv.gz" in dests
    assert "raw/gse273952/P_1/TLS_annotation.csv.gz" not in dests
    assert "raw/gse351843/P29_Vis_Rep2/filtered_feature_bc_matrix.h5" in dests
    assert "raw/gse351843/zenodo/pathology_annotations.zip" in dests
    assert "raw/usz/TLS_VISIUM_USZ.zip" in dests
    assert len(jobs) == 1 + 23 * 5 + 8 * 5 + 33 * 4 + 2


def test_download_skips_existing_file_without_network(tmp_path, monkeypatch):
    dest = tmp_path / "x.h5"
    dest.write_bytes(b"done")

    def no_network(url):
        raise AssertionError("existing files must not hit the server")

    monkeypatch.setattr(io_visium, "_remote_size", no_network)

    assert io_visium.download("https://example.org/x.h5", dest) == "skip"


def test_breast_h5ad_is_rebuilt_when_the_aggregate_mapping_changes(tmp_path):
    import anndata as ad

    dest = tmp_path / "GSE351843_P01_Vis.h5ad"
    built = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    built.uns["aggregate_is_tls"] = True
    built.write_h5ad(dest)

    assert io_visium.needs_build(dest, "GSE351843", aggregate_is_tls=False)
    assert not io_visium.needs_build(dest, "GSE351843", aggregate_is_tls=True)
    assert not io_visium.needs_build(dest, "GSE175540", aggregate_is_tls=False)  # labels do not depend on the flag
    assert io_visium.needs_build(tmp_path / "missing.h5ad", "GSE175540", aggregate_is_tls=True)


def test_spot_obs_refuses_annotation_whose_barcodes_do_not_match():
    import pytest

    barcodes = pd.Index(["AAAC-1", "AAAG-1"])
    positions = pd.DataFrame({"barcode": ["AAAC-1", "AAAG-1"], "in_tissue": [1, 1], "array_row": [0, 1],
                              "array_col": [0, 1], "pxl_row": [0, 1], "pxl_col": [0, 1]})
    annotation = pd.DataFrame({"Barcode": ["AAAC", "AAAG"], "TLS_2_cat": ["TLS", "NO_TLS"]})  # no "-1" suffix

    with pytest.raises(ValueError, match="barcodes"):
        io_visium.spot_obs(barcodes, positions, annotation)


def test_geo_annotation_takes_the_second_column_whatever_its_name(tmp_path):
    # GSE175540 frozen_a_1 / a_17 / b_7 name the label column "TLS", the other sections "TLS_2_cat".
    section = {s.name: s for s in io_visium.gse_sections()}["frozen_a_1"]
    folder = tmp_path / "gse175540" / "frozen_a_1"
    folder.mkdir(parents=True)
    with gzip.open(folder / "TLS_annotation.csv.gz", "wt") as handle:
        handle.write("Barcode,TLS\nB1,NO_TLS\nB2,TLS\n")

    table = io_visium.annotation_table(section, tmp_path)

    assert table.iloc[:, 1].tolist() == ["NO_TLS", "TLS"]
