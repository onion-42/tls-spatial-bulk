"""Patient-grouped splits, CV folds and sample weights.

Bulk: OOD cohorts are locked ("ood"). Every other cohort: 80/20 by patient, stratified by TLS status;
only patients with a pathologist label can be in test (CNN labels are train-only).
Visium: ~25% of patients held out, stratified by data source and TLS presence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _stratified_pick(keys: pd.DataFrame, strata: list[str], test_frac: float, rng) -> set:
    """Pick ~test_frac of the rows of `keys` (one row per patient) inside every stratum."""
    picked = set()
    for _, group in keys.groupby(strata, sort=True):
        n_test = int(round(len(group) * test_frac))
        picked |= set(rng.permutation(group["patient"].to_numpy())[:n_test])
    return picked


def split_bulk(obs: pd.DataFrame, ood_cohorts, test_frac: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    is_ood = obs["cohort"].isin(ood_cohorts)
    inner = obs[~is_ood]
    patients = (
        inner.assign(is_path=inner["label_source"].eq("pathologist"))
        .groupby("patient")
        .agg(cohort=("cohort", "first"), tls=("tls_any", "max"), is_path=("is_path", "any"))
        .reset_index()
    )
    eligible = patients[patients["is_path"]]
    test_patients = _stratified_pick(eligible, ["cohort", "tls"], test_frac, rng)
    split = pd.Series("train", index=obs.index)
    split[obs["patient"].isin(test_patients)] = "test"
    split[is_ood] = "ood"
    return split


def split_visium(patients: pd.DataFrame, test_frac: float, seed: int) -> pd.Series:
    """patients: one row per patient with columns patient, source, has_tls. Returns split per row."""
    rng = np.random.default_rng(seed)
    test_patients = _stratified_pick(patients, ["source", "has_tls"], test_frac, rng)
    return pd.Series(np.where(patients["patient"].isin(test_patients), "test", "train"), index=patients.index)


def cv_folds(groups: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    folds = np.full(len(groups), -1)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(groups)), y, groups)):
        folds[test_idx] = fold
    return folds


def balanced_weights(kind: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Every kind (bulk / visium) gets equal total weight; inside a kind every group does. Mean weight = 1."""
    frame = pd.DataFrame({"kind": kind, "group": group})
    n_groups = frame.groupby("kind")["group"].transform("nunique")
    group_size = frame.groupby(["kind", "group"])["group"].transform("size")
    w = 1.0 / (n_groups * group_size)
    return (w / w.mean()).to_numpy()
