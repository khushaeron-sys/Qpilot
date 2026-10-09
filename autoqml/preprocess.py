"""Encoding, splitting, dimensionality reduction. Owner: ML/Classical dev."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from .schemas import QuantumPlan

MAX_TEST = 100            # cap test rows so quantum inference stays fast
MAX_CATEGORIES = 20       # drop ID-like / very high-cardinality categoricals


def encode_frame(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Numeric + one-hot features, constants/IDs dropped, rows with missing target dropped."""
    df = df[df[target].notna()].copy()
    y = df[target]
    X = df.drop(columns=[target]).dropna(axis=1, how="all")
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols and X[c].nunique() <= MAX_CATEGORIES]
    parts = [X[num_cols].astype(float)]
    if cat_cols:
        parts.append(pd.get_dummies(X[cat_cols].astype(str), drop_first=True, dtype=float))
    Xe = pd.concat(parts, axis=1)
    Xe = Xe.loc[:, Xe.nunique(dropna=True) > 1]      # drop constant columns
    return Xe, y


def _safe_strat(y: np.ndarray) -> Optional[np.ndarray]:
    return y if np.bincount(y).min() >= 2 else None


def _subsample(X, y, n, seed):
    if n >= len(y):
        return X, y
    try:
        Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    except ValueError:
        Xs, _, ys, _ = train_test_split(X, y, train_size=n, random_state=seed)
    return Xs, ys


@dataclass
class PreparedData:
    X_train_q: np.ndarray; y_train_q: np.ndarray      # reduced + scaled to [0, pi], subsampled
    X_test_q: np.ndarray;  y_test: np.ndarray
    X_train_full: np.ndarray; y_train_full: np.ndarray  # all encoded features, all train rows (may contain NaN)
    X_test_full: np.ndarray
    class_names: List[str] = field(default_factory=list)
    feature_names_q: List[str] = field(default_factory=list)
    explained_variance: Optional[float] = None


def prepare_data(df: pd.DataFrame, target: str, plan: QuantumPlan) -> PreparedData:
    Xe, y_raw = encode_frame(df, target)
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    X = Xe.values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=_safe_strat(y), random_state=plan.seed)
    if len(y_te) > MAX_TEST:
        X_te, y_te = _subsample(X_te, y_te, MAX_TEST, plan.seed)

    X_trq, y_trq = _subsample(X_tr, y_tr, plan.train_subsample, plan.seed)

    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if plan.dim_reduction == "pca":
        steps.append(("red", PCA(n_components=plan.n_qubits, random_state=plan.seed)))
    elif plan.dim_reduction == "select_k_best":
        steps.append(("red", SelectKBest(f_classif, k=plan.n_qubits)))
    steps.append(("mm", MinMaxScaler(feature_range=(0.0, np.pi))))   # angle encoding range
    pipe = Pipeline(steps).fit(X_trq, y_trq)

    Xq_tr = pipe.transform(X_trq)
    Xq_te = np.clip(pipe.transform(X_te), 0.0, np.pi)
    assert Xq_tr.shape[1] == plan.n_qubits, f"expected {plan.n_qubits} features, got {Xq_tr.shape[1]}"

    names = list(Xe.columns)
    ev = None
    if plan.dim_reduction == "pca":
        names = [f"PC{i+1}" for i in range(plan.n_qubits)]
        ev = float(pipe.named_steps["red"].explained_variance_ratio_.sum())
    elif plan.dim_reduction == "select_k_best":
        names = [n for n, keep in zip(names, pipe.named_steps["red"].get_support()) if keep]

    return PreparedData(Xq_tr, y_trq, Xq_te, y_te, X_tr, y_tr, X_te,
                        class_names=list(le.classes_), feature_names_q=names, explained_variance=ev)
