"""MAB agents: zero-shot LLM, scheduled temperature, DORA (lambda + scoring).

Heavy imports (llm, dora_lambda_schedule) are deferred so that
``python run.py baselines`` works without a GPU.
"""


def __getattr__(name):
    _lazy = {
        "HF_MODEL": (".llm", "HF_MODEL"),
        "LLMBanditAgent": (".llm", "LLMBanditAgent"),
        "parse_bandit_color_strict": (".llm", "parse_bandit_color_strict"),
        "query_llm": (".llm", "query_llm"),
        "ScheduledTempLLMAgent": (".scheduled_temp", "ScheduledTempLLMAgent"),
        "LambdaPolicyLLMAgent": (".dora_lambda_schedule", "LambdaPolicyLLMAgent"),
    }
    if name in _lazy:
        import importlib
        mod_path, attr = _lazy[name]
        mod = importlib.import_module(mod_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HF_MODEL",
    "LLMBanditAgent",
    "LambdaPolicyLLMAgent",
    "ScheduledTempLLMAgent",
    "parse_bandit_color_strict",
    "query_llm",
]
