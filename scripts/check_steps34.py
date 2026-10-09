"""Quick verification script for Steps 3-4 (profiler + preprocess)."""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from autoqml.profiler import profile_dataset
from autoqml.preprocess import prepare_data
from autoqml.schemas import QuantumPlan, FeatureMapCfg, AnsatzCfg, OptimizerCfg

df = pd.read_csv(Path(__file__).parents[1] / "data" / "breast_cancer.csv")
print("Loaded breast_cancer.csv:", df.shape)

prof = profile_dataset(df, "target")
print("Profile:")
print("  n_rows=%d n_features_raw=%d n_features_encoded=%d n_classes=%d" % (
    prof.n_rows, prof.n_features_raw, prof.n_features_encoded, prof.n_classes))
print("  missing_pct=%.2f" % prof.missing_pct)

plan = QuantumPlan(
    rationale="test",
    task_type="binary_classification",
    n_qubits=4,
    dim_reduction="pca",
    feature_map=FeatureMapCfg(name="ZFeatureMap", reps=1, entanglement=None),
    ansatz=AnsatzCfg(name="EfficientSU2", reps=2, entanglement="linear"),
    optimizer=OptimizerCfg(name="COBYLA", maxiter=30),
    train_subsample=80,
    classical_baseline="logistic_regression",
    seed=42,
)
data = prepare_data(df, "target", plan)
print("PreparedData:")
print("  X_train_q shape:", data.X_train_q.shape)
print("  X_test_q shape:", data.X_test_q.shape)
print("  X_train_full shape:", data.X_train_full.shape)
print("  y classes:", data.class_names)
print("  feature_names_q:", data.feature_names_q)
print("  X_train_q range: [%.4f, %.4f]" % (data.X_train_q.min(), data.X_train_q.max()))
assert data.X_train_q.shape[1] == plan.n_qubits
assert 0.0 <= data.X_train_q.min()
assert data.X_train_q.max() <= 3.15  # ~ pi
print("Steps 3-4: PASS")
