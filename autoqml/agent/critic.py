import json
from typing import List, Optional
from ..fallback import parse_qubit_cap
from ..schemas import Attempt, DatasetProfile, Review
from .llm import call_tool, tool_schema
from .planner_prompt import CRITIC_SYSTEM_PROMPT
from .validator import validate_plan

REVIEW_TOOL = {
    "name": "submit_review",
    "description": "Accept the current result or propose ONE revised plan. This is the ONLY allowed way to respond.",
    "input_schema": tool_schema(Review),
}


def _summ(a: Attempt, i: int) -> dict:
    h = a.qnn.loss_history
    return {
        "attempt": i + 1,
        "plan": a.plan.model_dump(exclude={"rationale"}),
        "qnn": {"train_acc": round(a.qnn.train_accuracy, 3), "test_acc": round(a.qnn.test_accuracy, 3),
                "loss_first": round(h[0], 3) if h else None, "loss_last": round(h[-1], 3) if h else None,
                "n_params": a.qnn.n_params, "cx_count": a.qnn.cx_count, "seconds": a.qnn.train_seconds},
        "baseline": {"model": a.baseline.name, "matched_acc": round(a.baseline.matched_accuracy, 3),
                     "full_acc": round(a.baseline.full_accuracy, 3)},
        "evaluation": a.evaluation.model_dump(),
    }


def critique(user_prompt: str, profile: DatasetProfile, attempts: List[Attempt],
             attempts_left: int, call=call_tool) -> Optional[Review]:
    msg = (f"<dataset_profile>\n{profile.model_dump_json()}\n</dataset_profile>\n"
           f"<user_request>\n{user_prompt}\n</user_request>\n"
           f"<attempts>\n{json.dumps([_summ(a, i) for i, a in enumerate(attempts)], indent=1)}\n</attempts>\n"
           f"Attempts left after this one: {attempts_left}. Call submit_review.")
    try:
        _, raw, _ = call(CRITIC_SYSTEM_PROMPT, [{"role": "user", "content": msg}], REVIEW_TOOL)
        review = Review.model_validate(raw)
    except Exception:
        return None                                              # critic failure == accept
    if review.action == "revise":
        tried = [a.plan.model_dump(exclude={"rationale"}) for a in attempts]
        p = review.revised_plan
        if p is None or validate_plan(p, profile, parse_qubit_cap(user_prompt)) or p.model_dump(exclude={"rationale"}) in tried:
            return Review(action="accept", diagnosis=review.diagnosis + " (proposed revision rejected by validator)")
    return review
