"""Thin Claude wrapper: forced tool call => schema-shaped JSON, never free text. Owner: Agent dev."""
import os
import anthropic

MODEL = os.getenv("AUTOQML_MODEL", "claude-sonnet-5-5")
_client = None


def tool_schema(model_cls) -> dict:
    """Pydantic -> self-contained JSON Schema (inline $ref/$defs, drop titles)."""
    schema = model_cls.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node
    return resolve(schema)


def call_tool(system: str, messages: list, tool: dict, max_tokens: int = 1500):
    """Returns (tool_use_id, tool_input_dict, raw_content_blocks)."""
    global _client
    _client = _client or anthropic.Anthropic()           # reads ANTHROPIC_API_KEY
    resp = _client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        temperature=0,                                    # remove this line if your model rejects it
        system=system, messages=messages, tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return block.id, block.input, resp.content
