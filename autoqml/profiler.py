"""Dataset profiling (deterministic, no LLM). Owner: ML/Classical dev."""
import pandas as pd
from .preprocess import encode_frame
from .schemas import ColumnInfo, DatasetProfile

_TARGET_HINTS = {"target", "label", "class", "y", "diagnosis", "outcome", "output", "condition", "num", "status"}


def guess_target(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in _TARGET_HINTS:
            return c
    return df.columns[-1]


def profile_dataset(df: pd.DataFrame, target: str) -> DatasetProfile:
    Xe, y = encode_frame(df, target)
    cols = []
    for c in df.columns:
        s = df[c]
        kind = "boolean" if pd.api.types.is_bool_dtype(s) else "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
        cols.append(ColumnInfo(name=str(c), kind=kind, n_unique=int(s.nunique()), n_missing=int(s.isna().sum())))
    bal = y.astype(str).value_counts(normalize=True)
    return DatasetProfile(
        n_rows=int(len(y)),
        n_features_raw=int(df.shape[1] - 1),
        n_features_encoded=int(Xe.shape[1]),
        target_column=str(target),
        n_classes=int(y.nunique()),
        class_balance={str(k): round(float(v), 3) for k, v in bal.items()},
        missing_pct=round(float(df.isna().mean().mean() * 100), 2),
        columns=cols[:40],                       # cap tokens sent to the LLM
    )
