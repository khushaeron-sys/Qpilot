"""Classical baselines. Owner: ML/Classical dev."""
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .preprocess import PreparedData
from .schemas import BaselineResult, QuantumPlan


def _model(name: str, seed: int):
    return {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=800, random_state=seed),
    }[name]()


def _fit_eval(name, seed, Xtr, ytr, Xte, yte):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), _model(name, seed))
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xte)
    return float(accuracy_score(yte, p)), float(f1_score(yte, p, average="macro"))


def run_baselines(data: PreparedData, plan: QuantumPlan) -> BaselineResult:
    t0 = time.time()
    # 1) MATCHED: exactly what the QNN saw (same reduced features, same training rows)
    m_acc, m_f1 = _fit_eval(plan.classical_baseline, plan.seed, data.X_train_q, data.y_train_q, data.X_test_q, data.y_test)
    # 2) FULL: all features, all training rows  -> honest "best classical" reference
    f_acc, f_f1 = _fit_eval(plan.classical_baseline, plan.seed, data.X_train_full, data.y_train_full, data.X_test_full, data.y_test)
    return BaselineResult(
        name=plan.classical_baseline,
        matched_accuracy=m_acc, matched_f1_macro=m_f1, matched_n_features=int(data.X_train_q.shape[1]),
        full_accuracy=f_acc, full_f1_macro=f_f1, full_n_features=int(data.X_train_full.shape[1]),
        train_seconds=round(time.time() - t0, 2),
    )
