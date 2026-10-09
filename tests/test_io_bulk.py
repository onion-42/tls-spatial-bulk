import gzip

import pandas as pd

from tls_model import io_bulk

# A cohort name the repo has never seen (e.g. an archive-side name): must land in private.
UNKNOWN_COHORT = "Archive_lung"


def _label_table():
    return pd.DataFrame(
        {
            "cohort": ["TCGA-LUAD", UNKNOWN_COHORT, "MGH_LUAD_DagogoJack", "TCGA-KIRC/KIRP/KICH"],
            "sample_id": ["T1-01", "INT000001", "MGH1", "K1-01"],
            "patient_id": ["T1", "INTP1", "MGHP1", "K1"],
            "expression_id": ["T1-01", "INT000001", "MGH1", "K1-01"],
            "has_expression": [True, True, True, False],
            "method": ["pathologist_HE", "HookNet-TLS_b", "pathologist_HE", "pathologist_HE"],
            "source": ["internal/projects/x.tsv"] * 4,
            "notes": ["EXCLUDED by pathologist: yes; pathologist: DS", "pathologist: AG", None, None],
            "expression_file": ["expressions/a.tsv.gz"] * 4,
            "TLS_status": ["positive", "negative", "positive", "negative"],
            "maturity_status": ["unknown"] * 4,
        }
    )


def test_sanitize_labels_splits_private_and_drops_internal_columns():
    public, private = io_bulk.sanitize_labels(_label_table())

    assert set(public["cohort"]) == {"TCGA-LUAD", "TCGA-KIRC/KIRP/KICH"}
    assert set(private["cohort"]) == {UNKNOWN_COHORT, "MGH_LUAD_DagogoJack"}
    for table in (public, private):
        assert not {"source", "notes", "expression_file"} & set(table.columns)
    assert public.set_index("sample_id")["excluded"].to_dict() == {"T1-01": True, "K1-01": False}


def test_only_listed_public_cohorts_are_public():
    assert io_bulk.is_public_cohort("TCGA-KIRC/KIRP/KICH") and io_bulk.is_public_cohort("TCGA-KIRC_KIRP_KICH")
    assert not io_bulk.is_public_cohort(UNKNOWN_COHORT)
    assert not io_bulk.is_public_cohort("MGH_LUAD_DagogoJack")


def test_split_signatures_keeps_only_paper_and_chess_public(tmp_path):
    path = tmp_path / "sig.tsv"
    path.write_text(
        "name\tpriority\tinternal, CHESS or paper?\tpaper\thow\twhat\tprog\tspec\tGMT\t\t\n"
        "TLS_NL\t1\tinternal\t-\t-\t-\t-\t-\tCD86\tSELL\t\n"
        "TLS_Chem\t1\tpaper\thttps://doi.org/x\t-\t-\t-\t-\tCCL19\tCCL21\tCXCL13\n"
        "Treg\t0\tCHESS\t-\t-\t-\t-\t-\tFOXP3\t\t\n"
        "Breg\t\t\t-\t-\t-\t-\t-\tIL10\t\t\n",
        encoding="utf-8",
    )

    public, private = io_bulk.split_signatures(path)

    assert public["name"].tolist() == ["TLS_Chem", "Treg"]
    assert public.set_index("name").loc["TLS_Chem", "genes"] == "CCL19 CCL21 CXCL13"
    assert private["name"].tolist() == ["TLS_NL", "Breg"]


def test_cohort_stem_is_filename_safe():
    assert io_bulk.cohort_stem("TCGA-KIRC/KIRP/KICH") == "TCGA-KIRC_KIRP_KICH"


def _write_matrix(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, index=["G1", "G2"]).T
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, sep="\t")


def test_load_bulk_joins_labels_drops_excluded_and_unexpressed(tmp_path):
    public_dir, private_dir = tmp_path / "bulk", tmp_path / "private"
    public, private = io_bulk.sanitize_labels(_label_table())
    (public_dir).mkdir()
    (private_dir).mkdir()
    public.to_csv(public_dir / io_bulk.LABELS_PRIMARY, sep="\t", index=False)
    private.to_csv(private_dir / io_bulk.LABELS_PRIMARY, sep="\t", index=False)
    _write_matrix(public_dir / "expressions" / "TCGA-LUAD__log2TPM.tsv.gz", {"T1-01": [1.0, 2.0]})
    _write_matrix(private_dir / "expressions" / f"{UNKNOWN_COHORT}__log2TPM.tsv.gz", {"INT000001": [3.0, 4.0]})
    _write_matrix(private_dir / "expressions" / "MGH_LUAD_DagogoJack__log2TPM.tsv.gz", {"MGH1": [5.0, 6.0]})

    adata = io_bulk.load_bulk(public_dir, private_dir)

    # T1-01 is EXCLUDED, K1-01 has no expression
    assert sorted(adata.obs_names) == ["INT000001", "MGH1"]
    assert list(adata.var_names) == ["G1", "G2"]
    assert adata["MGH1"].X.tolist() == [[5.0, 6.0]]
    assert adata.obs.loc["MGH1", "tls_any"] == 1
    assert adata.obs.loc["MGH1", "patient"] == "MGH_LUAD_DagogoJack:MGHP1"
