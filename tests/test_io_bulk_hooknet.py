import json

import numpy as np
import pandas as pd
import pytest

from tls_model import io_bulk_hooknet, labels

XML_HEAD = '<?xml version="1.0" ?>\n<ASAP_Annotations>\n    <Annotations>\n'
XML_TAIL = "    </Annotations>\n    <AnnotationGroups />\n</ASAP_Annotations>\n"


def _polygon(i: int, group: str) -> str:
    return (f'        <Annotation Name="Annotation {i}" Type="Polygon" PartOfGroup="{group}" Color="#FF5733">\n'
            '            <Coordinates><Coordinate Order="0" X="1" Y="1"/></Coordinates>\n        </Annotation>\n')


@pytest.fixture
def zenodo_dir(tmp_path):
    """Three cohorts, one patient with two slides (LUSC), one GC-free TLS slide (BLCA), one TLS-free slide (KIRC)."""
    slides = {
        "LUSC": {"TCGA-AA-0001-01Z-00-DX1": ["tls", "tls", "other"],
                 "TCGA-AA-0001-01Z-00-DX2": ["tls", "gc", "lymph node"]},
        "BLCA": {"TCGA-BB-0002-01Z-00-DX1": ["tls", "tls", "tls", "inflammation"]},
        "KIRC": {"TCGA-CC-0003-01Z-00-DX1": ["other", "tumor"]},
    }
    xml_dir = tmp_path / "annotations_approx_05um"
    xml_dir.mkdir()
    for cohort, by_slide in slides.items():
        for slide, groups in by_slide.items():
            body = "".join(_polygon(i, g) for i, g in enumerate(groups))
            (xml_dir / f"{slide}.xml").write_text(XML_HEAD + body + XML_TAIL, encoding="utf-8")
    index = {cohort: [f"{s}.xml" for s in by_slide] for cohort, by_slide in slides.items()}
    (tmp_path / "annotations.json").write_text(json.dumps(index), encoding="utf-8")
    return tmp_path


def test_hooknet_labels_gc_is_mature(zenodo_dir):
    table = io_bulk_hooknet.hooknet_labels(zenodo_dir).set_index("sample_id")

    assert table.loc["TCGA-AA-0001-01", "maturity_status"] == labels.MATURE
    assert table.loc["TCGA-BB-0002-01", "maturity_status"] == labels.IMMATURE
    assert table.loc["TCGA-CC-0003-01", "TLS_status"] == "negative"
    assert table.loc["TCGA-CC-0003-01", "maturity_status"] == "unknown"
    assert set(table["method"]) == {io_bulk_hooknet.METHOD}
    assert table["cohort"].to_dict() == {"TCGA-AA-0001-01": "TCGA-LUSC", "TCGA-BB-0002-01": "TCGA-BLCA",
                                         "TCGA-CC-0003-01": "TCGA-KIRC/KIRP/KICH"}
    assert table.loc["TCGA-AA-0001-01", "patient_id"] == "TCGA-AA-0001"


def test_hooknet_two_slides_take_max(zenodo_dir):
    table = io_bulk_hooknet.hooknet_labels(zenodo_dir).set_index("sample_id")

    assert table.loc["TCGA-AA-0001-01", "TLS_count"] == 2  # max(2, 1), not the sum
    assert table.loc["TCGA-AA-0001-01", "TLS_mature_count"] == 1
    assert table.loc["TCGA-AA-0001-01", "n_slides"] == 2
    assert len(table) == 3


def test_hooknet_strict_label(zenodo_dir):
    table = io_bulk_hooknet.hooknet_labels(zenodo_dir).set_index("sample_id")

    assert table["TLS_status"].to_dict() == {"TCGA-AA-0001-01": "positive", "TCGA-BB-0002-01": "positive",
                                            "TCGA-CC-0003-01": "negative"}
    assert table["TLS_status_strict"].to_dict() == {"TCGA-AA-0001-01": "positive", "TCGA-BB-0002-01": "negative",
                                                   "TCGA-CC-0003-01": "negative"}


def test_hooknet_manual_labels_count_as_expert_labels(zenodo_dir):
    table = io_bulk_hooknet.hooknet_labels(zenodo_dir)

    harmonised = labels.bulk_labels(table).set_index(table["sample_id"])

    assert set(harmonised["label_source"]) == {"pathologist"}
    assert harmonised["tls_any"].to_dict() == {"TCGA-AA-0001-01": 1.0, "TCGA-BB-0002-01": 1.0, "TCGA-CC-0003-01": 0.0}
    assert harmonised.loc["TCGA-AA-0001-01", "tls_mature"] == 1.0
    assert harmonised.loc["TCGA-BB-0002-01", "tls_mature"] == 0.0
    assert np.isnan(harmonised.loc["TCGA-CC-0003-01", "tls_mature"])


def test_xena_matrix_reindexed_to_archive_genes():
    xena = pd.DataFrame(
        {"TCGA-XX-0001-01A": [1.0, 2.0, 3.0, 4.0], "TCGA-XX-0001-01B": [9.0, 9.0, 9.0, 9.0],
         "TCGA-XX-0002-11A": [5.0, 5.0, 5.0, 5.0], "TCGA-XX-0003-01A": [0.5, 1.5, 2.5, 3.5]},
        index=["ENSG1.1", "ENSG2.2", "ENSG3.1", "ENSG4.3"],
    )
    # ENSG4.3 is only listed by its version-free id and with a _PAR_Y copy (GENCODE pseudo-autosomal entries).
    probemap = pd.DataFrame({"id": ["ENSG1.1", "ENSG2.2", "ENSG3.1", "ENSG4.2", "ENSG4.2_PAR_Y"],
                             "gene": ["A", "B", "B", "Z", "Z"]})
    archive_genes = ["A", "B", "C"]

    out = io_bulk_hooknet.xena_to_archive(xena, probemap, archive_genes)

    assert list(out.columns) == archive_genes
    assert list(out.index) == ["TCGA-XX-0001-01", "TCGA-XX-0003-01"]  # primary tumour only, one vial per sample
    assert out.loc["TCGA-XX-0001-01"].tolist() == [1.0, 3.0, 0.0]  # duplicate symbol -> max; absent gene -> 0
    assert out.index.name == "expression_id"
