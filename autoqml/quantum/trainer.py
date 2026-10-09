"""Train a real PQC classifier (Qiskit ML VQC) on a simulator. Owner: Quantum dev."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from qiskit_machine_learning.algorithms import VQC

try:
    from qiskit_machine_learning.utils import algorithm_globals
except ImportError:
    try:
        from qiskit_algorithms.utils import algorithm_globals
    except ImportError:
        algorithm_globals = None

from ..preprocess import PreparedData
from ..schemas import QNNResult, QuantumPlan
from .builders import build_ansatz, build_feature_map, build_optimizer, circuit_stats


def get_sampler(kind: str = "reference", seed: int = 42):
    """Return the best available Qiskit primitive sampler.

    Tries V2 StatevectorSampler first (exact, deterministic, preferred by
    qiskit-machine-learning 0.8+), then falls back gracefully.
    """
    if kind == "aer":
        try:
            from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
            s = AerSamplerV2()
            s.options.default_shots = 2048
            return s
        except (ImportError, AttributeError):
            from qiskit_aer.primitives import Sampler as AerSampler
            return AerSampler(run_options={"shots": 2048, "seed": seed})

    # Prefer exact V2 StatevectorSampler (Qiskit 1.x standard)
    try:
        from qiskit.primitives import StatevectorSampler
        try:
            return StatevectorSampler(seed=seed)
        except TypeError:
            return StatevectorSampler()
    except ImportError:
        pass

    # Final fallback: V1 Sampler (still present in Qiskit 1.4.x)
    from qiskit.primitives import Sampler as V1Sampler
    try:
        return V1Sampler(options={"seed": seed})
    except TypeError:
        return V1Sampler()


def _make_loss_tracking_minimizer(base_optimizer, losses: list):
    """Wrap any qiskit_algorithms optimizer to capture per-evaluation losses.

    qiskit_algorithms COBYLA uses scipy under the hood which does NOT fire
    the VQC callback after each iteration. This wrapper instruments the
    objective function itself so every function evaluation is captured,
    giving us a full loss curve regardless of optimizer type.

    Returns a plain callable that satisfies the VQC Minimizer protocol.
    """
    def tracked_minimize(fun, x0, jac=None, bounds=None):
        def tracked_fun(x):
            val = fun(x)
            losses.append(float(val))
            return val

        return base_optimizer.minimize(fun=tracked_fun, x0=x0, jac=jac, bounds=bounds)

    return tracked_minimize


def _labels(pred) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:      # one-hot → argmax
        return pred.argmax(axis=1)
    return pred.ravel().astype(int)


def train_qnn(plan: QuantumPlan, data: PreparedData, sampler_kind: str = "reference") -> QNNResult:
    if algorithm_globals is not None:
        algorithm_globals.random_seed = plan.seed

    fm = build_feature_map(plan.feature_map, plan.n_qubits)
    an = build_ansatz(plan.ansatz, plan.n_qubits)
    init = np.random.default_rng(plan.seed).uniform(-np.pi, np.pi, an.num_parameters)

    losses: list = []
    base_opt = build_optimizer(plan.optimizer)

    # Use a tracked minimizer so losses are populated for ALL optimizer types
    # (COBYLA via scipy does not fire the VQC callback; SPSA does, but we
    #  capture both uniformly this way).
    tracked_min = _make_loss_tracking_minimizer(base_opt, losses)

    clf = VQC(
        feature_map=fm,
        ansatz=an,
        optimizer=tracked_min,          # callable Minimizer protocol
        loss="cross_entropy",
        sampler=get_sampler(sampler_kind, plan.seed),
        initial_point=init,
    )
    t0 = time.time()
    clf.fit(data.X_train_q, data.y_train_q)
    secs = time.time() - t0

    p_tr = _labels(clf.predict(data.X_train_q))
    p_te = _labels(clf.predict(data.X_test_q))
    stats = circuit_stats(plan)
    return QNNResult(
        train_accuracy=float(accuracy_score(data.y_train_q, p_tr)),
        test_accuracy=float(accuracy_score(data.y_test, p_te)),
        test_f1_macro=float(f1_score(data.y_test, p_te, average="macro")),
        loss_history=losses,
        n_params=stats["n_params"],
        n_qubits=plan.n_qubits,
        circuit_depth=stats["depth"],
        cx_count=stats["cx_count"],
        train_seconds=round(secs, 2),
        weights=[float(w) for w in clf.weights],
    )
