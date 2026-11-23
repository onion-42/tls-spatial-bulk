"""Model ladder for one head (P(any TLS), P(mature | TLS), or the direct 3-class no TLS / immature / mature),
in-fold feature selection, calibration, SHAP.

Families:
    signature   logistic regression on ssGSEA scores of public signatures (pre-specified baseline)
    prev_panel  elastic-net on a fixed external gene panel (the previous model's genes)
    ktsp        k top-scoring pairs: votes of "gene i < gene j" rules (rank-only classifier)
    enet        elastic-net logistic regression on the selected panel
    lgbm        LightGBM on the selected panel
    tabpfn      TabPFN (tabular foundation model, in-context learning) on the selected panel; binary heads only;
                needs the optional dependency group `tabpfn` and a GPU for more than ~5,000 rows
Selection happens inside `fit`, i.e. inside every CV fold. Selectors that score columns:
    corr   weighted |correlation| of the column with the label
    shap   mean |SHAP| of an auxiliary LightGBM (whatever the family)
    coef   mean |coefficient| of an auxiliary elastic-net on standardised columns (the linear model's own ranking)
    score all base columns -> top `n_pair_genes` signature genes -> add all their pairwise products
    -> re-score base top-`n_panel` + products -> keep top `n_panel`.
Selector `all` selects nothing: the panel is every base column, without pair products (for a model that weighs
the columns itself, e.g. TabPFN).
With three classes in y (enet, lgbm only) `corr` scores a column by its best one-vs-rest correlation, `shap` by the
mean |SHAP| over classes, and `predict_proba` returns the (n, 3) matrix.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tls_model.features import pair_products, weighted_abs_corr

FAMILIES = ("signature", "prev_panel", "ktsp", "enet", "lgbm", "tabpfn")
# How the sample weights reach TabPFN: `native` = fit(sample_weight=w) (experimental upstream, raises when the
# checkpoint takes no row weights); `resample` = bootstrap the fit rows with probability proportional to w.
TABPFN_WEIGHTS = ("native", "resample")
SELECTORS = ("corr", "shap", "coef", "all")
LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1,
)
SHAP_SELECTOR_PARAMS = {**LGBM_PARAMS, "n_estimators": 200, "colsample_bytree": 0.3}
# Elastic-net logistic regression via SGD: saga needed ~200 s per fit on ~22k units.
ENET_PARAMS = dict(loss="log_loss", penalty="elasticnet", l1_ratio=0.5, alpha=1e-3, max_iter=50, tol=1e-4, average=True)
EPS = 1e-6


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def three_class(p_any: np.ndarray, p_mature: np.ndarray) -> np.ndarray:
    """Columns: no TLS, immature TLS, mature TLS."""
    return np.column_stack([1 - p_any, p_any * (1 - p_mature), p_any * p_mature])


def _lgbm_shap(model: LGBMClassifier, X: np.ndarray) -> np.ndarray:
    """Per-class SHAP contributions without the bias column: (n, n_classes, n_features); n_classes = 1 for binary."""
    contrib = model.predict(X, pred_contrib=True)
    n_classes = contrib.shape[1] // (X.shape[1] + 1)
    return contrib.reshape(X.shape[0], n_classes, X.shape[1] + 1)[:, :, :-1]


def weighted_resample(X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap sample of the rows drawn with probability proportional to `w`: turns sample weights into row
    frequencies for a model without weights. A class with positive weight that the draw missed gets its heaviest
    row appended, so no class disappears (the result is then longer than the input by the number of such classes).
    One draw only: the fitted model carries this resampling noise."""
    w = np.asarray(w, dtype=float)
    if np.any(w < 0) or not w.sum() > 0:
        raise ValueError("sample weights must be non-negative with a positive sum")
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(y), size=len(y), replace=True, p=w / w.sum())
    drawn = set(np.unique(y[rows]))
    rescue = []
    for c in np.unique(y):
        candidates = np.flatnonzero(y == c)
        if c not in drawn and w[candidates].sum() > 0:
            rescue.append(candidates[np.argmax(w[candidates])])
    rows = np.concatenate([rows, np.array(rescue, dtype=int)])
    return X[rows], y[rows]


class KTSP:
    """k disjoint top-scoring pairs (Geman et al. 2004), weighted by sample weights."""

    def __init__(self, k: int = 7):
        self.k = k
        self.pairs: list[tuple[int, int, bool]] = []

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> "KTSP":
        w1 = np.where(y == 1, w, 0.0) / w[y == 1].sum()
        w0 = np.where(y == 0, w, 0.0) / w[y == 0].sum()
        n_genes = X.shape[1]
        scores = np.zeros((n_genes, n_genes))
        for i in range(n_genes):
            less = (X[:, [i]] < X).astype(np.float64)
            scores[i] = w1 @ less - w0 @ less
        upper = np.triu(np.abs(scores), k=1)
        used, pairs = set(), []
        for flat in np.argsort(-upper, axis=None):
            i, j = divmod(int(flat), n_genes)
            if len(pairs) >= min(self.k, n_genes // 2) or upper[i, j] <= 0:
                break
            if i in used or j in used:
                continue
            pairs.append((i, j, bool(scores[i, j] > 0)))
            used |= {i, j}
        self.pairs = pairs
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.pairs:
            return np.full(X.shape[0], 0.5)
        votes = [(X[:, i] < X[:, j]) if up else (X[:, i] >= X[:, j]) for i, j, up in self.pairs]
        return np.mean(votes, axis=0)


class HeadModel:
    def __init__(
        self,
        family: str,
        selector: str | None = None,
        n_panel: int = 200,
        n_pair_genes: int = 30,
        signature_genes: frozenset | set = frozenset(),
        panel_genes: list[str] | None = None,
        seed: int = 0,
        n_shap_rows: int = 3000,
        preselected: tuple[list[str], list[str]] | None = None,
        tabpfn_weights: str = "resample",
    ):
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}")
        if family in ("ktsp", "enet", "lgbm", "tabpfn") and selector not in SELECTORS:
            raise ValueError(f"{family} needs a selector in {SELECTORS}")
        if tabpfn_weights not in TABPFN_WEIGHTS:
            raise ValueError(f"tabpfn_weights must be one of {TABPFN_WEIGHTS}, got {tabpfn_weights!r}")
        self.tabpfn_weights = tabpfn_weights
        self.family, self.selector = family, selector
        self.n_panel, self.n_pair_genes = n_panel, n_pair_genes
        self.signature_genes = set(signature_genes)
        self.panel_genes = panel_genes or []
        self.seed, self.n_shap_rows = seed, n_shap_rows
        self.panel: list[str] = []
        self.pair_genes: list[str] = []
        self.preselected = preselected  # (panel, pair_genes) shared by enet/lgbm within one fold
        self.model = None
        self.n_classes = 2

    def _assemble(self, B: np.ndarray, names: list[str], columns: list[str]) -> np.ndarray:
        index = {n: i for i, n in enumerate(names)}
        out = np.empty((B.shape[0], len(columns)), dtype=np.float32)
        for k, col in enumerate(columns):
            if col.startswith("genepair:"):
                left, right = col.split(":", 1)[1].split("*")
                out[:, k] = B[:, index[f"gene:{left}"]] * B[:, index[f"gene:{right}"]]
            else:
                out[:, k] = B[:, index[col]]
        return out

    def _score(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self.selector == "corr":
            classes = np.unique(y)
            if len(classes) <= 2:
                return weighted_abs_corr(X, y.astype(float), w)
            return np.max([weighted_abs_corr(X, (y == c).astype(float), w) for c in classes], axis=0)
        if self.selector == "coef":
            linear = make_pipeline(StandardScaler(), SGDClassifier(random_state=self.seed, **ENET_PARAMS))
            linear.fit(X, y, sgdclassifier__sample_weight=w)
            return np.abs(linear[-1].coef_).mean(axis=0)
        model = LGBMClassifier(random_state=self.seed, **SHAP_SELECTOR_PARAMS).fit(X, y, sample_weight=w)
        rng = np.random.default_rng(self.seed)
        rows = rng.choice(len(X), size=min(self.n_shap_rows, len(X)), replace=False)
        return np.abs(_lgbm_shap(model, X[rows])).mean(axis=(0, 1))

    def _select(self, B: np.ndarray, names: list[str], y: np.ndarray, w: np.ndarray) -> list[str]:
        if self.family == "signature":
            return [n for n in names if n.startswith("sig:")]
        if self.family == "prev_panel":
            present = set(names)
            return [f"gene:{g}" for g in self.panel_genes if f"gene:{g}" in present]
        if self.family == "ktsp":
            gene_idx = [i for i, n in enumerate(names) if n.startswith("gene:")]
            scores = self._score(B[:, gene_idx], y, w)
            return [names[gene_idx[i]] for i in np.argsort(-scores)[: self.n_panel]]
        if self.selector == "all":
            return list(names)

        scores = self._score(B, y, w)
        order = np.argsort(-scores)
        sig_cols = [i for i in order if names[i].startswith("gene:") and names[i][5:] in self.signature_genes]
        top_sig = sig_cols[: self.n_pair_genes]
        self.pair_genes = [names[i][5:] for i in top_sig]
        pairs, pair_names = pair_products(B[:, top_sig], self.pair_genes, prefix="genepair")
        top_base = list(order[: self.n_panel])
        union = np.hstack([B[:, top_base], pairs])
        union_names = [names[i] for i in top_base] + pair_names
        union_scores = self._score(union, y, w)
        return [union_names[i] for i in np.argsort(-union_scores)[: self.n_panel]]

    def fit(self, B: np.ndarray, names: list[str], y: np.ndarray, w: np.ndarray) -> "HeadModel":
        y = np.asarray(y).astype(int)
        self.n_classes = len(np.unique(y))
        if self.n_classes > 2 and self.family in ("ktsp", "signature", "prev_panel", "tabpfn"):
            raise ValueError(f"{self.family} is binary only; use enet or lgbm for {self.n_classes} classes")
        if self.preselected is not None:
            self.panel, self.pair_genes = list(self.preselected[0]), list(self.preselected[1])
        else:
            self.panel = self._select(B, names, y, w)
        X = self._assemble(B, names, self.panel)
        if self.family == "lgbm":
            self.model = LGBMClassifier(random_state=self.seed, **LGBM_PARAMS).fit(X, y, sample_weight=w)
        elif self.family == "ktsp":
            self.model = KTSP().fit(X, y, w)
        elif self.family == "tabpfn":
            self.model = self._fit_tabpfn(X, y, w)
        else:
            if self.family == "signature":
                linear = LogisticRegression(C=1.0, max_iter=2000)
            else:
                linear = SGDClassifier(random_state=self.seed, **ENET_PARAMS)
            self.model = make_pipeline(StandardScaler(), linear)
            self.model.fit(X, y, **{f"{self.model.steps[-1][0]}__sample_weight": w})
        return self

    def _fit_tabpfn(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        from tabpfn import TabPFNClassifier  # optional dependency group `tabpfn` (pulls in torch)

        model = TabPFNClassifier(random_state=self.seed, fit_mode="fit_with_cache")
        if self.tabpfn_weights == "native":
            return model.fit(X, y, sample_weight=w)
        return model.fit(*weighted_resample(X, y, w, self.seed))

    def predict_proba(self, B: np.ndarray, names: list[str]) -> np.ndarray:
        """P(class 1) for a binary head, the (n, n_classes) matrix for a multiclass one."""
        X = self._assemble(B, names, self.panel)
        if self.family == "ktsp":
            return self.model.predict_proba(X)
        probs = self.model.predict_proba(X)
        return probs if self.n_classes > 2 else probs[:, 1]

    def shap_values(self, B: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], np.ndarray] | None:
        """Per-unit SHAP values (log-odds scale) of the panel features; None for kTSP and TabPFN.
        For a multiclass head: the contributions to the last class (mature)."""
        X = self._assemble(B, names, self.panel)
        if self.family in ("ktsp", "tabpfn"):
            return None
        if self.family == "lgbm":
            return _lgbm_shap(self.model, X)[:, -1, :], list(self.panel), X
        scaler, linear = self.model[0], self.model[-1]
        values = (X - scaler.mean_) / scaler.scale_ * linear.coef_[-1]
        return values, list(self.panel), X


class PlattCalibrator:
    """Logistic recalibration of out-of-fold probabilities, one per unit kind (bulk / spot).

    `prior` = {kind: true prevalence} when the fitting rows were class-enriched (e.g. sampled Visium spots):
    the calibrated logit is shifted by logit(true) − logit(sampled) (Saerens et al., 2002)."""

    def __init__(self):
        self.models: dict[str, LogisticRegression] = {}
        self.shift: dict[str, float] = {}

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p: np.ndarray, y: np.ndarray, kind: np.ndarray, prior: dict[str, float] | None = None) -> "PlattCalibrator":
        prior = prior or {}
        for k in np.unique(kind):
            mask = kind == k
            if len(np.unique(y[mask])) == 2:
                self.models[k] = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p[mask]), y[mask])
                if k in prior:
                    sampled = float(np.mean(y[mask]))
                    self.shift[k] = float(self._logit(np.array([prior[k]]))[0, 0] - self._logit(np.array([sampled]))[0, 0])
        return self

    def transform(self, p: np.ndarray, kind: np.ndarray) -> np.ndarray:
        out = np.asarray(p, dtype=float).copy()
        for k, model in self.models.items():
            mask = kind == k
            if mask.any():
                q = model.predict_proba(self._logit(p[mask]))[:, 1]
                if k in self.shift:
                    q = 1.0 / (1.0 + np.exp(-(self._logit(q)[:, 0] + self.shift[k])))
                out[mask] = q
        return out
