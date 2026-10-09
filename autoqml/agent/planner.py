from pydantic import ValidationError
from ..fallback import default_plan, parse_qubit_cap
from ..schemas import DatasetProfile, QuantumPlan
from .llm import call_tool, tool_schema
from .planner_prompt import PLANNER_SYSTEM_PROMPT
from .validator import validate_plan

PLAN_TOOL = {
    "name": "submit_quantum_plan",
    "description": "Submit the final quantum ML architecture plan. This is the ONLY allowed way to respond.",
    "input_schema": tool_schema(QuantumPlan),
}


def build_user_message(user_prompt: str, profile: DatasetProfile) -> str:
    return (f"<dataset_profile>\n{profile.model_dump_json(indent=1)}\n</dataset_profile>\n\n"
            f"<user_request>\n{user_prompt}\n</user_request>\n\nCall submit_quantum_plan.")


def plan_with_retries(user_prompt: str, profile: DatasetProfile, max_retries: int = 2, call=call_tool):
    """Claude proposes -> code validates -> errors fed back -> deterministic fallback as last resort."""
    cap = parse_qubit_cap(user_prompt)
    messages = [{"role": "user", "content": build_user_message(user_prompt, profile)}]
    last_errors = []
    for attempt in range(1, max_retries + 2):
        try:
            tool_id, raw, content = call(PLANNER_SYSTEM_PROMPT, messages, PLAN_TOOL)
        except Exception as e:                                   # no API key / network / rate limit
            return default_plan(profile, user_prompt), {"source": "fallback", "reason": f"LLM unavailable: {type(e).__name__}"}
        try:
            plan = QuantumPlan.model_validate(raw)
            errors = validate_plan(plan, profile, cap)
        except ValidationError as e:
            plan, errors = None, [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()]
        if plan is not None and not errors:
            return plan, {"source": "claude", "attempts": attempt}
        last_errors = errors
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "is_error": True,
                "content": "Plan rejected by validator:\n- " + "\n- ".join(errors) +
                           "\nCall submit_quantum_plan again with a corrected plan."}]},
        ]
    return default_plan(profile, user_prompt), {"source": "fallback", "reason": "validation failed", "errors": last_errors}
