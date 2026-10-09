"""The agent loop: Profile -> Plan -> Train -> Compare -> Critique -> (Revise) -> Report. Owner: Agent dev."""
import pandas as pd
from ..baseline import run_baselines
from ..catalog import MAX_CLASSES
from ..evaluator import evaluate
from ..preprocess import prepare_data
from ..profiler import profile_dataset
from ..quantum.trainer import train_qnn
from ..schemas import Attempt, RunReport, RunRequest
from .critic import critique
from .llm import call_tool
from .planner import plan_with_retries


def run_autoqml(df: pd.DataFrame, request: RunRequest, max_iterations: int = 2,
                sampler_kind: str = "reference", on_event=None, call=call_tool) -> RunReport:
    emit = on_event or (lambda kind, payload: None)

    profile = profile_dataset(df, request.target_column)
    emit("profile", profile)
    if not (2 <= profile.n_classes <= MAX_CLASSES):
        raise ValueError(f"Target '{request.target_column}' has {profile.n_classes} distinct values; "
                         f"AutoQML supports classification with 2-{MAX_CLASSES} classes.")

    plan, meta = plan_with_retries(request.user_prompt, profile, call=call)
    emit("plan", (plan, meta))

    attempts = []
    for i in range(max_iterations):
        data = prepare_data(df, request.target_column, plan)
        emit("training", plan)
        qnn = train_qnn(plan, data, sampler_kind)
        base = run_baselines(data, plan)
        att = Attempt(plan=plan, qnn=qnn, baseline=base, evaluation=evaluate(qnn, base))
        attempts.append(att)
        emit("trained", att)

        left = max_iterations - 1 - i
        if left == 0:
            break
        review = critique(request.user_prompt, profile, attempts, left, call=call)
        if review is None or review.action == "accept":
            att.critic_note = review.diagnosis if review else "Critic unavailable; accepted."
            emit("critic", att.critic_note)
            break
        att.critic_note = review.diagnosis
        emit("critic", f"Revising: {review.diagnosis}")
        plan = review.revised_plan

    return RunReport(request=request, profile=profile, planner_meta=meta, attempts=attempts)
