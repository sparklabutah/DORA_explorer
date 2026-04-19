"""Tree-of-thought style agent: one LLM call with structured multi-step reasoning (k=3, b=1)."""

import argparse
import copy
import re

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
    messages2conversation,
)


SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)



SIMPLE_DECISION_PROMPT = (
    "\n\n---\n"
    "## Tree of Thought Reasoning\n\n"

    "### Branch Generation (k=3)\n"
    "Consider the current game state and generate EXACTLY 3 distinct candidate actions.\n"
    "Each branch must differ meaningfully — not just rephrasings of the same idea.\n"
    "Format:\n"
    "  Branch A: <action>\n"
    "  Branch B: <action>\n"
    "  Branch C: <action>\n\n"

    "### Branch Evaluation\n"
    "For each branch, reason through ONE step ahead: what is the likely outcome?\n"
    "Score each on two dimensions (1–5 each):\n"
    "  - Progress: Does it advance toward the goal or unlock new options?\n"
    "  - Safety: Does it avoid irreversible mistakes or dead ends?\n"
    "Format:\n"
    "  Branch A: Progress=? Safety=? Total=? | Reasoning: <one sentence>\n"
    "  Branch B: Progress=? Safety=? Total=? | Reasoning: <one sentence>\n"
    "  Branch C: Progress=? Safety=? Total=? | Reasoning: <one sentence>\n\n"

    "### Pruning (b=1)\n"
    "Eliminate the two lower-scoring branches.\n"
    "State which branch survives and why in one sentence.\n\n"

    "### Final Action\n"
    "After pruning, output exactly ONE line in this format (this line must be last):\n"
    "ACTION: <single short game command to type, e.g. open fridge>\n\n"
    "Rules:\n"
    "- Put only the command after ACTION: (no backticks, no quotes).\n"
    "- Do not write ACTION: anywhere except this final line.\n"
    "- If unsure what to type, use: ACTION: help\n"
)


def _is_likely_meta_or_score_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    low = s.lower()
    if s.startswith(("#", "---", "```", "* ", "- ", "• ")):
        return True
    if re.match(r"^#{1,6}\s", s):
        return True
    if re.match(r"^branch\s+[abc]:", low):
        return True
    if "progress=" in low or "safety=" in low or "| reasoning:" in low:
        return True
    if low.startswith(("format:", "step ", "rules:", "example:")):
        return True
    return False


def _parse_action_from_tot_response(raw: str) -> str:
    """Extract the game command from ToT output; prefer a final ACTION: line (CoT-style)."""
    text = raw.strip()
    if not text:
        return ""

    # Strip optional trailing ``` fence
    if text.endswith("```"):
        text = text[: text.rfind("```")].rstrip()

    # 1) Last ACTION: line wins (same convention as llm_cot)
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        m = re.match(r"^ACTION:\s*(.+)$", stripped, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("`\"'")

    # 2) Explicit final labels
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        m = re.match(
            r"^(?:Final|Winning)\s+(?:Action|Command)\s*:\s*(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().strip("`\"'")

    # 3) Last plausible command line (skip markdown / branch / score lines)
    candidates = []
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped or _is_likely_meta_or_score_line(stripped):
            continue
        # One line of plain command text
        if len(stripped) > 200:
            continue
        if re.search(r"^[\s\d.)\-–—]+\s*", stripped):
            stripped = re.sub(r"^[\s\d.)\-–—]+\s*", "", stripped).strip()
        candidates.append(stripped)
        if len(candidates) >= 1:
            break

    if candidates:
        return candidates[0].strip("`\"'")

    # 4) Fallback: last non-empty line
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip().strip("`\"'")

    return text


class LLMToTAgent(tales.Agent):
    """Single-call agent with implicit ToT (k=3, b=1)."""

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.conversation = kwargs["conversation"]
        self.tot_max_tokens = kwargs["tot_max_tokens"]

    @property
    def uid(self):
        return (
            f"LLMToTAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_tt{self.tot_max_tokens}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "tot",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
            "tot_max_tokens": self.tot_max_tokens,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def _apply_llm_compat(self, llm_kwargs: dict) -> None:
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
            "claude-3.7-sonnet",
        ]:
            llm_kwargs.pop("seed", None)

        if "gemini" in self.llm or "gemma" in self.llm:
            llm_kwargs.pop("seed", None)
            if "max_tokens" in llm_kwargs:
                llm_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        messages = merge_messages(messages)

        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

        return messages

    def act(self, obs, reward, done, infos):
        obs_line = f"{obs}\n> "
        messages = self.build_messages(obs_line)

        messages_simple = copy.deepcopy(messages)
        messages_simple[-1] = {
            **messages_simple[-1],
            "content": messages_simple[-1]["content"] + SIMPLE_DECISION_PROMPT,
        }

        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": self.tot_max_tokens,
            "seed": self.seed,
            "stream": False,
        }
        self._apply_llm_compat(llm_kwargs)

        response = self._llm_call_from_messages(messages_simple, **llm_kwargs)
        raw = response.text()
        action = _parse_action_from_tot_response(raw)

        self.history.append((obs_line, f"{action}\n"))

        stats = {
            "prompt": format_messages_to_markdown(messages_simple),
            "response": raw,
            "nb_tokens_prompt": self.token_counter(messages=messages_simple),
            "nb_tokens_response": self.token_counter(text=raw),
        }
        stats["nb_tokens"] = stats["nb_tokens_prompt"] + stats["nb_tokens_response"]

        return action, stats


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMToTAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--tot-max-tokens",
        type=int,
        default=512,
        help="Max tokens for the ToT completion (reasoning + final command). Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="tot",
    desc="Single-call agent with structured ToT reasoning (k=3, b=1).",
    klass=LLMToTAgent,
    add_arguments=build_argparser,
)
