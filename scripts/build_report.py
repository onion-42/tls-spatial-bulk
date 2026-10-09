"""Build the report from results/ (PNG figures embedded as base64) + per-sample inputs of report_data.py.

report.html       public: the per-sample UMAP shows only cohorts whose data-use terms allow it (RND_USE = Allowed).
report_full.html  local (gitignored): every cohort.
Usage: uv run python scripts/build_report.py [--full]   (from the repo root)
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
RES, FIG, ICI = ROOT / "results", ROOT / "results" / "figures", ROOT / "results" / "ici"
IS_FULL = "--full" in sys.argv[1:]
OUT = ROOT / ("report_full.html" if IS_FULL else "report.html")
PRIVATE = ROOT / "data" / "private" / "report"  # per-sample inputs from report_data.py (no sample ids)
OURS = "TLS model (ours)"


def tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def ci(est, lo, hi, digits=3) -> str:
    if pd.isna(est):
        return "–"
    return f"{est:.{digits}f} <span class=ci>[{lo:.{digits}f}, {hi:.{digits}f}]</span>"


def table(df: pd.DataFrame, caption: str = "", index: bool = False) -> str:
    """DataFrame -> HTML. Cells are escaped here unless they were built by ci() / mark()."""
    def cell(v):
        if isinstance(v, str) and ("<span" in v or "<b>" in v):
            return v
        if isinstance(v, float):
            return "–" if pd.isna(v) else (f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}")
        return html.escape(str(v))

    df = df.reset_index() if index else df
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class=tw><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(name: str, caption: str) -> str:
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}" loading="lazy">'
            f"<figcaption><b>{name}</b> · {caption}</figcaption></figure>")


def metric_rows(df: pd.DataFrame, head: str) -> pd.DataFrame:
    t = df[df["head"] == head]
    return pd.DataFrame({
        "evaluation": t["split"], "block": t["block"], "n": t["n"].astype("Int64"), "TLS+": t["n_pos"].astype("Int64"),
        "AUROC": [ci(*r) for r in t[["auroc", "auroc_lo", "auroc_hi"]].itertuples(index=False)],
        "AUPRC": [ci(*r) for r in t[["auprc", "auprc_lo", "auprc_hi"]].itertuples(index=False)],
        "Brier": [ci(*r) for r in t[["brier", "brier_lo", "brier_hi"]].itertuples(index=False)],
    })


def effect_pivot(assoc: pd.DataFrame) -> pd.DataFrame:
    """HR / OR per 1 SD [95% CI] per endpoint × diagnosis × score; bold when the CI excludes 1."""
    a = assoc[assoc["score"] != "covariates only"].copy()

    def mark(r):
        if pd.isna(r.estimate):
            return "–"
        text = ci(r.estimate, r.lo, r.hi, 2)
        return f"<b>{text}</b>" if (r.lo > 1 or r.hi < 1) else text

    a["cell"] = a.apply(mark, axis=1)
    order = {e: i for i, e in enumerate(["OS", "PFS", "Response", "DCR"])}
    p = a.pivot_table(index=["endpoint", "diagnosis"], columns="score", values="cell", aggfunc="first")
    p = p.reindex(sorted(p.index, key=lambda k: (order[k[0]], k[1] == "All", k[1])))
    return p[[OURS] + [c for c in p.columns if c != OURS]]


def delta_pivot(delta: pd.DataFrame, rows, column: str, values_from: str) -> pd.DataFrame:
    d = delta[rows].copy()

    def mark(r):
        if pd.isna(r.delta):
            return "–"
        text = f"{r.delta:+.3f} <span class=ci>[{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]</span>"
        return f"<b>{text}</b>" if (r.delta_lo > 0 or r.delta_hi < 0) else text

    d["cell"] = d.apply(mark, axis=1)
    order = {e: i for i, e in enumerate(["OS", "PFS", "Response", "DCR"])}
    p = d.pivot_table(index=["endpoint", "diagnosis"], columns=column, values="cell", aggfunc="first")
    return p.reindex(sorted(p.index, key=lambda k: (order[k[0]], k[1] == "All", k[1])))


def added_pivot(delta: pd.DataFrame) -> pd.DataFrame:
    d = delta.copy()

    def mark(r):
        if pd.isna(r.delta):
            return "–"
        p = "" if pd.isna(r.lr_p) else f" · p={r.lr_p:.2g}"
        text = f"{r.delta:+.3f} <span class=ci>[{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]{p}</span>"
        return f"<b>{text}</b>" if (pd.notna(r.lr_p) and r.lr_p < 0.05) else text

    d["cell"] = d.apply(mark, axis=1)
    d["comparison"] = d["comparison"].str.split(":").str[0]
    order = {e: i for i, e in enumerate(["OS", "PFS", "Response", "DCR"])}
    p = d.pivot_table(index=["endpoint", "diagnosis"], columns="comparison", values="cell", aggfunc="first")
    p = p.reindex(sorted(p.index, key=lambda k: (order[k[0]], k[1] == "All", k[1])))
    return p[[c for c in ["a", "b", "b vs a", "c", "d", "e", "e vs d"] if c in p.columns]]


def interactive_roc() -> str:
    path = PRIVATE / "roc.json"
    if not path.exists():
        return "<p class=pending>Interactive ROC: run scripts/report_data.py first.</p>"
    return f"""
<h3>ROC by cohort (interactive)</h3>
<p>Pathologist TLS labels on held-out patients and the two OOD cohorts; our final model against the three published
signatures (ssGSEA on bulk; on spots the signatures use only the model's gene panel). Pick a cohort; hover a curve
for its AUROC.</p>
<div class=controls><label for=roc-block>Cohort</label><select id=roc-block></select></div>
<div id=roc-plot class=plot></div>
<p>Our model leads on Visium spots, on the Internal lung OOD cohort and on bulk pooled across cohorts. Inside single
held-out bulk cohorts (17–45 patients each) a published signature is often as good or better, e.g. the 12-chemokine
signature in TCGA-PAAD and CPTAC-ccRCC; these cohorts are small and their intervals wide.</p>
<script type="application/json" id=roc-data>{path.read_text(encoding="utf-8")}</script>"""


def interactive_umap() -> str:
    path = PRIVATE / "umap.tsv"
    if not path.exists():
        return "<p class=pending>Interactive UMAP: run scripts/report_data.py first.</p>"
    u = pd.read_csv(path, sep="\t")
    shown = u if IS_FULL else u[u["rnd_use"] == "Allowed"]
    scope = ("all ICI-treated samples" if IS_FULL else
             f"{len(shown):,} of {len(u):,} ICI-treated samples: studies whose data-use terms do not allow sharing "
             "per-patient values are left out here (they are in every aggregate table above)")
    shown = shown.drop(columns="rnd_use")
    return f"""
<h3>UMAP of ICI-treated samples (interactive)</h3>
<p>Shown: {scope}. Embedding computed on all samples: log2(TPM+1) of the 2,000 most variable genes, standardised,
50 principal components, UMAP. Samples group mainly by study; colour by cohort shows the batch structure that
within-cohort z-scoring and cohort strata remove from the outcome models.</p>
<div class=controls><label for=umap-colour>Colour by</label><select id=umap-colour>
<option value=diagnosis>diagnosis</option><option value=cohort>cohort</option><option value=response>response (ORR)</option>
<option value=p_tls>P(TLS), our model</option><option value=chem>12-chemokine score</option></select></div>
<div id=umap-plot class="plot tall"></div>
<script type="application/json" id=umap-data>{json.dumps(shown.to_dict("list"))}</script>"""


def main() -> None:
    test, ood = tsv(RES / "test_metrics.tsv"), tsv(RES / "ood_metrics.tsv")
    cv_a, cv_b, loco = tsv(RES / "cv_stage_a.tsv"), tsv(RES / "cv_stage_b_any.tsv"), tsv(RES / "loco.tsv")
    ceiling, cohorts_bulk = tsv(RES / "label_ceiling_luad.tsv"), tsv(RES / "bulk_cohorts.tsv")
    coverage, ici_cohorts = tsv(ICI / "coverage.tsv"), tsv(ICI / "cohorts.tsv")
    rho = tsv(ICI / "score_correlation.tsv")
    assoc_main, delta_main = tsv(ICI / "assoc_main.tsv"), tsv(ICI / "delta_main.tsv")
    assoc_n, delta_n = tsv(ICI / "assoc_tmb_pdl1.tsv"), tsv(ICI / "delta_tmb_pdl1.tsv")
    added = tsv(ICI / "delta_added_value.tsv")

    mature_all = test[(test["head"] == "mature") & (test["block"] == "bulk:all")].iloc[0]
    sig_cols = [c for c in rho.columns if c != "diagnosis"]
    cv_b_view = cv_b[["candidate", "auroc_bulk", "auroc_spot", "auroc_bulk_pooled", "score"]].rename(columns={
        "auroc_bulk": "bulk AUROC, within cohort", "auroc_spot": "spot AUROC", "auroc_bulk_pooled": "bulk AUROC, pooled",
        "score": "selection score"})
    cv_a_view = cv_a.rename(columns={"auroc_bulk": "bulk AUROC, within cohort", "auroc_spot": "spot AUROC",
                                     "auroc_bulk_pooled": "bulk AUROC, pooled", "score": "selection score"})

    sections = {
        "summary": ("Summary", f"""
<p class=lede>One expression-only model reads tertiary lymphoid structures (TLS) from a Visium spot or from a bulk
RNA-seq sample, one sample at a time. It was trained on spot annotations from spatial transcriptomics together with
pathologist TLS calls on bulk tumours, selected by grouped cross-validation, and checked once on two locked
out-of-distribution (OOD) cohorts. It was then frozen and scored in 2,689 patients treated with immune checkpoint
inhibitors (ICI).</p>
<ul class=findings>
<li><b>Spots work.</b> Held-out Visium patients: AUROC {ci(*test[(test['head']=='any')&(test['block']=='visium:spots')][['auroc','auroc_lo','auroc_hi']].iloc[0])}.</li>
<li><b>Bulk ranks TLS across cancer types, weakly within one.</b> Held-out bulk, pooled: AUROC
{ci(*test[(test['head']=='any')&(test['block']=='bulk:all')][['auroc','auroc_lo','auroc_hi']].iloc[0])}; within a cohort
0.45–0.70 with wide intervals. Every model family, published signatures included, stays near 0.55–0.65 here.</li>
<li><b>OOD, scored once.</b> Internal lung {ci(*ood[ood['block']=='bulk:Internal_lung'][['auroc','auroc_lo','auroc_hi']].iloc[0])},
CPTAC-PAAD {ci(*ood[ood['block']=='bulk:CPTAC-PAAD'][['auroc','auroc_lo','auroc_hi']].iloc[0])}.</li>
<li><b>Immunotherapy.</b> A higher score goes with longer survival and more responses, but the effect is small
(pooled OS HR 0.92 per SD), and the model is not better than the 12-chemokine signature: added on top of it, it
contributes nothing (likelihood-ratio tests near null). A positive interaction of the two scores appears in bladder
cancer (IMvigor210) and ccRCC; it is uncorrected for multiplicity and is a hypothesis.</li>
</ul>"""),

        "background": ("Background", """
<p>A TLS is an ectopic lymphoid organ that forms inside chronically inflamed tissue, tumours included: a B-cell
follicle (mature TLS carry a germinal centre and follicular dendritic cells), a T-cell zone and high endothelial
venules. In 540 tumours treated with anti-PD-1 / PD-L1, mature TLS predicted objective response (43.1% vs 19.6%),
PFS and OS independently of PD-L1 and CD8 density (Vanhersecke et al., <i>Nat Cancer</i> 2021,
<a href="https://doi.org/10.1038/s43018-021-00232-6">doi:10.1038/s43018-021-00232-6</a>).</p>
<p>TLS are scored on H&amp;E, multiplex immunofluorescence or spatial transcriptomics: accurate, but slow and costly.
Bulk RNA-seq is routine, yet bulk TLS signatures are imprecise against pathologist calls. This project trains one
model on both worlds so that the same model scores a spot or a bulk sample.</p>
<div class=pending><b>Figure 2 · TLS in solid cancers</b> — redrawn in BioRender from the 2023–2026 literature;
to be added.</div>"""),

        "data": ("Data", f"""
<p>Bulk cohorts and their roles. TLS+ counts are pathologist calls except TCGA-BRCA and part of TCGA-LUAD, where
HookNet-TLS labels (CNN) were used for training only; every test and OOD metric uses pathologist labels.</p>
{table(cohorts_bulk)}
<p>Visium: GSE175540 (ccRCC, 23 sections, FFPE and frozen; Meylan <i>Immunity</i> 2022) and USZ (8 FFPE kidney and lung
sections; Zenodo 14620362, CC-BY-4.0). Splits are by patient: about 25% of Visium patients and 20% of bulk patients
per cohort are held out. Non-public cohorts (Internal lung, MGH) stay outside the repository.</p>
<p>Label ceiling, TCGA-LUAD: two pathologists agree almost perfectly with each other (κ 0.96), while two runs of
the public HookNet-TLS model (HookNet-TLS_a, HookNet-TLS_b) agree with pathologists only moderately (κ 0.20–0.51)
and with each other poorly (κ 0.20). A model trained partly on these CNN labels cannot be expected to beat that
agreement.</p>
{table(ceiling)}"""),

        "methods": ("Methods", """
<h3>Model</h3>
<ul>
<li><b>Units.</b> A bulk sample, a Visium spot, an aggregate of k neighbouring spots (k = 1, 7, 19) and a whole-section
pseudo-bulk; all share one gene panel (16,788 genes detected on every platform).</li>
<li><b>Features.</b> Within-sample ranks (chosen in stage A over log2(TPM+1); the bulk↔Visium platform offset did not
help and was dropped), ssGSEA of 30 public signatures, their pairwise products and products of the top signature
genes chosen inside each fold.</li>
<li><b>Ladder.</b> Signature baseline, elastic-net logistic regression, k-top-scoring pairs, LightGBM; feature
selection by correlation or by mean |SHAP|, both inside folds.</li>
<li><b>Selection.</b> Grouped 5-fold CV over training patients; the score weighs mean within-cohort bulk AUROC (pooled
AUROC rewards telling cancer types apart) and spot AUROC. Winner: LightGBM with SHAP selection.</li>
<li><b>Calibration.</b> Platt scaling on out-of-fold predictions, separately for bulk and spot units; spot
probabilities corrected for the TLS-enriched sampling of training spots.</li>
<li><b>Evaluation.</b> Held-out patients; bootstrap 95% CI for every metric; OOD cohorts scored once, with nothing
tuned afterwards; leave-one-cohort-out reported for robustness only.</li>
</ul>
<h3>Signature scores (baselines)</h3>
<ul>
<li><b>Gene sets.</b> Cabrita TLS signature (Cabrita et al., Nature 2020; 9 genes: CD1D, CETP, LAT, SKAP1, CCR6, RBP5,
CD79B, EIF1AY, PTGDS). 12-chemokine (Coppola et al., Am J Pathol 2011; CXCL9, CXCL10, CXCL11, CXCL13, CCL2, CCL3, CCL4,
CCL5, CCL8, CCL18, CCL19, CCL21). TLS imprint (Meylan et al., Immunity 2022; 29 genes as listed in Nat Commun 2025,
PMC12059173; PTLP is not an HGNC symbol and is absent, so 28 genes are scored).</li>
<li><b>Method.</b> Single-sample GSEA (Barbie et al., Nature 2009), one sample at a time, no reference cohort. Genes of a
sample are ranked by expression (average ranks for ties); with rank r and weight w = r<sup>0.25</sup>, the score is the
weighted mean rank of the set's genes minus the mean rank of all other genes. This is the closed form of the ssGSEA
running sum, unnormalised (no rescaling to the range of a cohort). Genes of a set missing from the matrix are skipped.</li>
<li><b>Input.</b> log2(TPM+1). ICI cohorts: all 20,062 genes shared by the 28 studies. Bulk ROC: all 20,062 genes of
the bulk matrix. Visium spots: only the model's gene panel (5,049 genes), so spot-level signature scores are an
approximation.</li>
<li><b>Scale in outcome models.</b> Each score, ours included (ours on the logit scale of P(TLS)), is z-scored within
cohort before modelling, so effects read per 1 SD inside a study.</li>
<li>The same ssGSEA implementation computes the 30 signature features inside our model.</li>
</ul>
<h3>Immunotherapy validation</h3>
<ul>
<li><b>Cohorts.</b> 28 published or internal ICI studies; regimens containing an ICI only (ICI with TKI,
bevacizumab or chemotherapy kept; TKI / mTOR / EGFR-only arms dropped); duplicate samples removed.</li>
<li><b>Scores.</b> Our P(TLS) on the logit scale, and three published TLS signatures as baselines: Cabrita (Nature
2020), 12-chemokine (Coppola, AJP 2011) and the TLS imprint (Meylan, Immunity 2022; 29 genes), all ssGSEA on
log2(TPM+1). Every score is z-scored within cohort, so HR and OR are per 1 SD.</li>
<li><b>Endpoints.</b> OS and PFS (Cox, stratified by cohort; lifelines); objective response from RECIST (CR/PR vs
SD/PD; study-annotated R/NR only where RECIST is missing) and disease control (CR/PR/SD vs PD) by logistic
regression with cohort fixed effects (statsmodels).</li>
<li><b>Confounders</b> in every model: ICI class, age, sex, stage, site, location. A missing category is its own
level; missing age is median-imputed with a missing indicator; covariates without variation inside cohorts are
dropped.</li>
<li><b>Comparison.</b> ΔC-index (OS, PFS) and ΔAUC (response, DCR) between models on the same patients, with a paired
bootstrap: patients resampled within cohort, every model refitted, 20 replicates.</li>
<li><b>TMB and PD-L1.</b> TMB-high = ≥ 10 mut/Mb (studies reporting raw mutation counts excluded); PD-L1 = tumour-cell
IHC category TC0 / TC1 / TC2+ as an ordinal covariate. Pooled always; a diagnosis alone only with ≥ 30 OS events.</li>
<li><b>Added value over the 12-chemokine signature.</b> Nested models with the same confounders on complete cases.
<b>a</b> 12-chemokine + ours vs 12-chemokine; <b>b</b> 12-chemokine + ours + 12-chemokine × ours vs 12-chemokine,
and vs a. On the TMB + PD-L1 subset (both as covariates, never in an interaction): <b>c</b> ours over TMB + PD-L1;
<b>d</b> ours over TMB + PD-L1 + 12-chemokine; <b>e</b> with the 12-chemokine × ours term, vs the d base and vs d.
The primary statistic is the likelihood-ratio test of each nested pair; Δ and HR / OR are descriptive.</li>
</ul>"""),

        "course": ("Course of the work", """
<ol class=timeline>
<li><b>Plan.</b> One model with a TLS head and a maturity head; OOD cohorts locked before any training.</li>
<li><b>First run, three fixes.</b> The elastic-net solver was slow and over-shrunk (replaced by SGD). The maturity
head looked inverted on bulk because all mature positives fell into one or two folds (fixed by stratifying folds by
data block and label). Bulk model selection moved to within-cohort AUROC.</li>
<li><b>Stage A.</b> Rank features beat log-expression; the platform offset did not help.</li>
<li><b>Stage B.</b> LightGBM with SHAP selection had the best selection score; signature baselines were close.</li>
<li><b>Review.</b> Calibration by unit type, lymph-node spots removed before aggregation, unknown labels made
explicit; leave-one-cohort-out kept as a robustness table only.</li>
<li><b>Maturity head removed.</b> Bulk maturity showed no signal.</li>
<li><b>OOD scored once</b> and reported as is.</li>
<li><b>Immunotherapy.</b> Model frozen; 28 studies harmonised; baselines, Cox and logistic models, TMB + PD-L1 subset.</li>
<li><b>Added value.</b> Nested models with the 12-chemokine signature and its interaction with our score.</li>
</ol>"""),

        "model": ("Model results", f"""
<h3>Cross-validation</h3>
{table(cv_a_view, "Stage A · feature representation (elastic net, correlation selection)")}
{table(cv_b_view, "Stage B · model × selector; the highest selection score was kept")}
<h3>Held-out test and OOD</h3>
{table(metric_rows(pd.concat([test, ood]), "any"), "P(any TLS): held-out patients and the two OOD cohorts")}
<p>Bulk maturity (pooled held-out): AUROC {ci(mature_all.auroc, mature_all.auroc_lo, mature_all.auroc_hi)} — no
usable signal, which is why the maturity output was removed.</p>
{table(loco, "Leave-one-cohort-out (robustness only; selection saw these cohorts)")}
{interactive_roc()}
{figure("01_label_maps.png", "Visium spot labels on example sections.")}
{figure("02_bulk_pca.png", "Bulk cohorts, PCA of the shared gene panel: cohorts separate, so pooled AUROC is easy and within-cohort AUROC is the honest metric.")}
{figure("03_reliability_test.png", "Calibration on held-out patients.")}
{figure("03_shap_any.png", "SHAP of the final TLS model.")}
{figure("04_reliability_ood.png", "Calibration on the OOD cohorts: bulk probabilities over-call TLS on a new cohort.")}
{figure("04_spot_maps.png", "Predicted P(TLS) on held-out Visium sections.")}"""),

        "ici": ("Immunotherapy cohorts", f"""
{table(coverage, "Patients per diagnosis and endpoint coverage")}
{table(ici_cohorts, "Studies and cohorts")}
{table(rho, "Spearman correlation of our score with each signature, within diagnosis")}
{interactive_umap()}
<h3>Association with outcome</h3>
<p>HR (OS, PFS) or OR (response, DCR) per 1 SD, adjusted; bold when the 95% CI excludes 1. HR &lt; 1 means longer
survival, OR &gt; 1 more responses.</p>
{table(effect_pivot(assoc_main), index=True)}
<h3>Discrimination: ours vs each signature</h3>
<p>ΔC-index or ΔAUC of the model with our score minus the model with the signature, same patients; bold when the
bootstrap CI excludes 0.</p>
{table(delta_pivot(delta_main, delta_main['reference'] == OURS, 'other', 'delta'), index=True)}
<p>Each score vs the confounders-only model:</p>
{table(delta_pivot(delta_main, delta_main['other'] == 'covariates only', 'reference', 'delta'), index=True)}
{''.join(figure(f'05_forest_{e}.png', f'{e}: left, adjusted HR / OR per SD; right, Δ of ours vs each signature.') for e in ['OS', 'PFS', 'Response', 'DCR'])}
<h3>TMB-high + PD-L1 subset</h3>
<p>Patients with both a TMB call and PD-L1 IHC (271 BLCA, 106 NSCLC). Base model = confounders + TMB-high + PD-L1.</p>
{table(effect_pivot(assoc_n), index=True)}
{table(delta_pivot(delta_n, delta_n['other'] == 'covariates only', 'reference', 'delta'), "Each score over confounders + TMB-high + PD-L1", index=True)}
{''.join(figure(f'05_forest_tmb_pdl1_{e}.png', f'{e}, adjusted for TMB-high and PD-L1.') for e in ['OS', 'PFS', 'Response', 'DCR'] if (FIG / f'05_forest_tmb_pdl1_{e}.png').exists())}"""),

        "added": ("Added value over 12-chemokine", f"""
<p>Δ (bigger − smaller nested model) with bootstrap CI, then the likelihood-ratio p. Bold: p &lt; 0.05,
uncorrected. Comparisons a–e are defined in Methods.</p>
{table(added_pivot(added), index=True)}
<p>Our score adds nothing over the 12-chemokine signature (a, d). On the TMB + PD-L1 subset it adds to TMB and PD-L1
alone (c), mostly in bladder cancer. The interaction term (b vs a, e vs d) reaches p &lt; 0.05 in bladder cancer and
ccRCC OS: patients high on both scores fare better than either score predicts. The subset and the bladder cohort are
largely the same IMvigor210 patients, so these are not independent confirmations.</p>
{figure("05_added_value_chem.png", "Comparisons a and b, all patients.")}
{figure("05_added_value_tmb_pdl1_chem.png", "Comparisons c–e, TMB-high + PD-L1 subset.")}"""),

        "limits": ("Limitations", """
<ul>
<li>Bulk within one cohort is weak (AUROC about 0.55–0.65 for every model family, signatures included); bulk
probabilities over-call TLS on a new cohort, so use the ranking there.</li>
<li>No maturity output: the maturity head showed no signal.</li>
<li>For immunotherapy outcome the model is at best equal to the 12-chemokine signature.</li>
<li>Bootstrap uses 20 replicates: intervals are rough and occasionally do not contain the point estimate.</li>
<li>No correction for multiplicity across diagnoses × endpoints × comparisons; the interaction finding is a hypothesis.</li>
<li>Pooled PFS and the TMB + PD-L1 PFS models needed a small ridge penalty to converge; no likelihood-ratio test is
reported for them.</li>
<li>Confounder coverage is uneven: age, stage and site are missing for most studies; TMB and PD-L1 exist for two
diagnoses only.</li>
<li>Several studies carry restricted-use terms; this report holds aggregates only and is not for public release.</li>
</ul>"""),
    }

    toc = "".join(f'<li><a href="#{k}">{t}</a></li>' for k, (t, _) in sections.items())
    body = "".join(f'<section id="{k}"><h2>{t}</h2>{content}</section>' for k, (t, content) in sections.items())
    OUT.write_text(TEMPLATE.replace("{{TOC}}", toc).replace("{{BODY}}", body), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>TLS Model Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">
<style>
/* Layout: sticky contents rail beside one reading column; stains of H&E as the palette (haematoxylin ink, eosin accent). */
:root {
  --paper: #fbf9fa; --ink: #231d2b; --muted: #6b6274; --rule: #e6dfe7; --tint: #f3eef4;
  --hema: #3b3a8c; --eosin: #c2456b;
  --display: "Source Serif 4", Georgia, serif; --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #17141c; --ink: #ebe6ef; --muted: #a59cae; --rule: #332d3b; --tint: #211d28;
  --hema: #a9a8f0; --eosin: #f08aa8; color-scheme: dark } }
:root[data-theme="dark"] {
  --paper: #17141c; --ink: #ebe6ef; --muted: #a59cae; --rule: #332d3b; --tint: #211d28;
  --hema: #a9a8f0; --eosin: #f08aa8; color-scheme: dark }
* { box-sizing: border-box }
body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.6 var(--body); }
.wrap { display: grid; grid-template-columns: 13rem minmax(0, 1fr); gap: 3rem; max-width: 78rem; margin: 0 auto;
        padding-inline: 24px; padding-block: 2.5rem 4rem; }
nav { position: sticky; top: 1.5rem; align-self: start; font-size: .9rem; }
nav p { font: 600 .72rem/1 var(--body); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0 0 .8rem }
nav ol { list-style: none; margin: 0; padding: 0; display: grid; gap: .35rem; }
nav a { color: var(--ink); text-decoration: none; border-left: 2px solid var(--rule); padding-left: .7rem; display: block }
nav a:hover, nav a:focus-visible { border-color: var(--eosin); color: var(--hema); outline: none }
main { min-width: 0 }
header h1 { font: 650 2.4rem/1.15 var(--display); margin: 0 0 .4rem; text-wrap: balance; color: var(--hema) }
header p { color: var(--muted); margin: 0 0 2rem; font-size: .95rem }
section { border-top: 1px solid var(--rule); padding-block: 1.6rem; }
h2 { font: 650 1.55rem/1.25 var(--display); margin: 0 0 1rem; text-wrap: balance }
h3 { font: 500 1.15rem/1.3 var(--display); margin: 1.6rem 0 .6rem; color: var(--hema) }
p, li { max-width: 68ch }
.lede { font-size: 1.08rem }
.findings { padding-left: 1.1rem; display: grid; gap: .5rem }
.findings li::marker { color: var(--eosin) }
.timeline { padding-left: 1.4rem; display: grid; gap: .45rem }
.timeline li::marker { color: var(--eosin); font-weight: 600 }
a { color: var(--hema) }
.ci { color: var(--muted); font-size: .88em; white-space: nowrap }
b .ci { color: inherit }
.tw { overflow-x: auto; margin: 1rem 0 1.4rem; }
table { border-collapse: collapse; font: .82rem/1.4 var(--mono); font-variant-numeric: tabular-nums; }
caption { text-align: left; font: 600 .85rem var(--body); color: var(--muted); padding-bottom: .5rem }
th { text-align: left; font: 600 .76rem/1.3 var(--body); color: var(--muted); border-bottom: 1.5px solid var(--ink);
     padding: .35rem .7rem; vertical-align: bottom }
td { padding: .3rem .7rem; border-bottom: 1px solid var(--rule); white-space: nowrap; vertical-align: top }
tbody tr:hover td { background: var(--tint) }
td b { color: var(--eosin); font-weight: 600 }
figure { margin: 1.4rem 0 2rem; }
figure img { width: 100%; max-width: 100%; height: auto; background: #fff; border: 1px solid var(--rule); border-radius: 4px }
figcaption { font-size: .85rem; color: var(--muted); margin-top: .4rem; max-width: 80ch }
.pending { border: 1px dashed var(--eosin); padding: .8rem 1rem; border-radius: 4px; color: var(--muted); max-width: 68ch }
.controls { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; font-size: .9rem }
.controls label { font-weight: 600; color: var(--muted) }
select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: .25rem .5rem }
select:focus-visible { outline: 2px solid var(--eosin); outline-offset: 1px }
.plot { width: 100%; height: 440px; border: 1px solid var(--rule); border-radius: 4px; margin-bottom: 1.6rem }
.plot.tall { height: 560px }
@media (max-width: 860px) {
  .wrap { grid-template-columns: minmax(0, 1fr); gap: 1rem; padding-inline: 16px }
  nav { position: static } nav ol { grid-template-columns: repeat(auto-fill, minmax(10rem, 1fr)) }
  header h1 { font-size: 1.9rem }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto } }
html { scroll-behavior: smooth }
</style>
</head>
<body>
<div class="wrap">
<nav aria-label="Contents"><p>Contents</p><ol>{{TOC}}</ol></nav>
<main>
<header><h1>TLS Model Report</h1><p>One expression-only TLS model for Visium spots and bulk RNA-seq · validation on held-out,
out-of-distribution and immunotherapy cohorts · aggregates only</p></header>
{{BODY}}
</main>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<script>
(function () {
  if (!window.Plotly) {
    document.querySelectorAll(".plot").forEach(el => { el.textContent = "Interactive plots need an internet connection (plotly.js from cdnjs)."; });
    return;
  }
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const PALETTE = ["#3b3a8c", "#c2456b", "#2a9d8f", "#e9a23b", "#7a5195", "#5c8001", "#d1495b", "#00798c", "#8d6e63", "#6c757d"];
  const base = () => ({
    paper_bgcolor: css("--paper"), plot_bgcolor: css("--paper"),
    font: { family: css("--body"), color: css("--ink"), size: 12 },
    margin: { l: 50, r: 10, t: 10, b: 45 },
    xaxis: { gridcolor: css("--rule"), zerolinecolor: css("--rule") },
    yaxis: { gridcolor: css("--rule"), zerolinecolor: css("--rule") },
    legend: { orientation: "h", y: -0.18 },
  });
  const config = { responsive: true, displaylogo: false };
  const redraws = [];

  const rocData = document.getElementById("roc-data");
  if (rocData) {
    const curves = JSON.parse(rocData.textContent);
    const keyOf = c => c.split + " | " + c.block;
    const blocks = [...new Set(curves.map(keyOf))];
    const select = document.getElementById("roc-block");
    blocks.forEach(b => { const o = document.createElement("option"); o.value = b; o.textContent = b.replace(" | ", " · "); select.appendChild(o); });
    const drawRoc = () => {
      const chosen = curves.filter(c => keyOf(c) === select.value);
      const traces = chosen.map((c, i) => ({
        x: c.fpr, y: c.tpr, type: "scatter", mode: "lines", line: { width: i === 0 ? 3 : 1.6, color: PALETTE[i] },
        name: c.score + " · AUROC " + c.auc.toFixed(3),
        hovertemplate: c.score + "<br>AUROC " + c.auc.toFixed(3) + "<br>FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>",
      }));
      traces.push({ x: [0, 1], y: [0, 1], mode: "lines", line: { dash: "dot", color: css("--muted"), width: 1 }, hoverinfo: "skip", showlegend: false });
      const layout = base();
      layout.xaxis.title = "false positive rate"; layout.yaxis.title = "true positive rate";
      layout.xaxis.range = [0, 1]; layout.yaxis.range = [0, 1.02];
      layout.annotations = [{ x: 0.98, y: 0.04, xref: "x", yref: "y", xanchor: "right", showarrow: false,
        text: "n = " + chosen[0].n + " (" + chosen[0].n_pos + " TLS+)", font: { color: css("--muted") } }];
      Plotly.react("roc-plot", traces, layout, config);
    };
    select.addEventListener("change", drawRoc);
    redraws.push(drawRoc);
  }

  const umapData = document.getElementById("umap-data");
  if (umapData) {
    const d = JSON.parse(umapData.textContent);
    const select = document.getElementById("umap-colour");
    const hover = d.x.map((_, i) => d.diagnosis[i] + " · " + d.cohort[i] + "<br>response: " + d.response[i] +
      "<br>P(TLS) " + d.p_tls[i].toFixed(3) + " · 12-chem " + d.chem[i].toFixed(3));
    const drawUmap = () => {
      const key = select.value;
      const layout = base();
      layout.xaxis.title = "UMAP 1"; layout.yaxis.title = "UMAP 2";
      layout.xaxis.showticklabels = false; layout.yaxis.showticklabels = false;
      let traces;
      if (key === "p_tls" || key === "chem") {
        traces = [{ x: d.x, y: d.y, type: "scattergl", mode: "markers", text: hover, hovertemplate: "%{text}<extra></extra>", showlegend: false,
          marker: { size: 4, color: d[key], colorscale: "Viridis", showscale: true, colorbar: { title: key === "p_tls" ? "P(TLS)" : "12-chem" } } }];
      } else {
        const levels = [...new Set(d[key])].sort();
        const fixed = key === "response" ? { "CR/PR": "#2a9d8f", "SD/PD": "#c2456b", "not available": css("--rule") } : null;
        traces = levels.map((lv, j) => {
          const idx = [];
          d[key].forEach((v, i) => { if (v === lv) idx.push(i); });
          return { x: idx.map(i => d.x[i]), y: idx.map(i => d.y[i]), text: idx.map(i => hover[i]), name: lv + " (" + idx.length + ")",
            type: "scattergl", mode: "markers", hovertemplate: "%{text}<extra></extra>",
            marker: { size: 4, color: fixed ? fixed[lv]
              : levels.length > PALETTE.length ? "hsl(" + Math.round(360 * j / levels.length) + ",55%,52%)" : PALETTE[j] } };
        });
        if (key === "cohort") { layout.legend = { font: { size: 9 }, orientation: "v", x: 1.02, y: 1 }; layout.margin.r = 10; }
      }
      Plotly.react("umap-plot", traces, layout, config);
    };
    select.addEventListener("change", drawUmap);
    redraws.push(drawUmap);
  }

  const redraw = () => redraws.forEach(f => f());
  redraw();
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  if (mq.addEventListener) mq.addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
