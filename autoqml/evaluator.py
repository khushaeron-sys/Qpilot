"""Compare QNN vs classical, characterise the loss curve. Owner: ML/Classical dev."""
import numpy as np
from .schemas import BaselineResult, Evaluation, QNNResult


def loss_trend(losses, w: int = 10) -> str:
    if len(losses) < 2 * w:
        return "too_short"
    a = np.asarray(losses, dtype=float)
    prev, last = a[-2 * w:-w].mean(), a[-w:].mean()
    rel = (prev - last) / max(abs(prev), 1e-9)
    if rel > 0.02:
        return "still_decreasing"
    if np.std(np.diff(a[-3 * w:])) > 0.25 * max(abs(last), 1e-9) and rel < 0:
        return "noisy"
    return "plateaued"


def evaluate(qnn: QNNResult, base: BaselineResult) -> Evaluation:
    d_m = qnn.test_accuracy - base.matched_accuracy
    d_f = qnn.test_accuracy - base.full_accuracy
    if d_m >= 0.02:
        v = f"QNN beats the matched classical baseline by {d_m*100:.1f} pts."
    elif d_m > -0.03:
        v = f"QNN is on par with the matched classical baseline ({d_m*100:+.1f} pts)."
    else:
        v = f"QNN trails the matched classical baseline by {abs(d_m)*100:.1f} pts."
    if d_f < -0.03:
        v += f" The best classical model on all features is {abs(d_f)*100:.1f} pts ahead."
    return Evaluation(delta_vs_matched=d_m, delta_vs_full=d_f,
                      overfit_gap=qnn.train_accuracy - qnn.test_accuracy,
                      loss_trend=loss_trend(qnn.loss_history), verdict=v)
