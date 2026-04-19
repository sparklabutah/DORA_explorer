"""Chain-of-thought baseline: same as ``llm.LLMAgent`` but with CoT instructions and ACTION: parsing."""

import argparse
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


SYSTEM_PROMPT_COT = (
    "You are playing a text-based adventure game. Your goal is to finish it with the highest score.\n\n"
    "Before acting, reason briefly covering:\n"
    "  1. What the current scene tells you (exits, objects, state).\n"
    "  2. What your immediate sub-goal is.\n"
    "  3. Which single command best advances that sub-goal.\n\n"
    "After your reasoning, output exactly one line and nothing after it:\n"
    "ACTION: <single short command, e.g. get lamp>\n\n"
    "Rules:\n"
    "- Put only the game command after ACTION: (a few words). No second sentences on that line.\n"
    "- Do not write ACTION: anywhere except this final line.\n"
    "- No backticks or quotes around the command.\n"
    "When stuck, try using the 'help' command to see what commands are available."
)

_MAX_CMD_LEN = 200


def _strip_cmd_wrapping(s: str) -> str:
    s = s.strip()
    s = s.strip("`\"'")
    return s.strip()


def _normalize_cot_command(cmd: str) -> str:
    """Single-line, bounded-length command for the game and for history."""
    if not cmd:
        return "help"
    one = cmd.split("\n")[0].strip()
    one = re.sub(r"\s+", " ", one)
    one = _strip_cmd_wrapping(one)
    # Drop trailing period/comma if they look like sentence punctuation, not decimals
    one = re.sub(r"[\s.]+$", "", one)
    if len(one) > _MAX_CMD_LEN:
        one = one[:_MAX_CMD_LEN].rsplit(" ", 1)[0] if " " in one else one[:_MAX_CMD_LEN]
    return one or "help"


def _line_looks_like_prose_line(s: str) -> bool:
    """Heuristic: likely narration, not a text-game command."""
    t = s.strip()
    if len(t) > 160:
        return True
    if t.count(" ") > 35:
        return True
    low = t.lower()
    if low.startswith(
        (
            "therefore",
            "thus,",
            "so,",
            "in conclusion",
            "i think",
            "i will",
            "i should",
            "we should",
            "the player",
            "looking at",
        )
    ):
        return True
    if len(t.split()) > 12:
        return True
    return False


def _parse_action_from_cot_response(raw: str) -> str:
    """Return only the game command; never the full CoT transcript as the action."""
    text = (raw or "").strip()
    if not text:
        return "help"

    # Drop trailing ``` fence if the model wrapped the end of the reply
    if "```" in text:
        text = re.sub(r"```\s*$", "", text).rstrip()

    # 1) Last line that matches ACTION: ... (preferred; scan bottom-up)
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        stripped = re.sub(r"^\*+\s*", "", stripped)
        m = re.match(r"^ACTION:\s*(.*)$", stripped, flags=re.IGNORECASE)
        if m:
            cmd = m.group(1).strip()
            if not cmd:
                continue
            cmd = _normalize_cot_command(cmd)
            if cmd:
                return cmd

    # 2) Inline ACTION: on a line (model sometimes prefixes with markdown / filler)
    for line in reversed(text.splitlines()):
        m = re.search(r"\bACTION:\s*([^\n]+)", line.strip(), flags=re.IGNORECASE)
        if m:
            cmd = _normalize_cot_command(m.group(1))
            if cmd:
                return cmd

    # 3) Fallback: last short, non-prose line (avoid dumping full reasoning)
    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s or s.startswith(("#", "```", "---", "**")):
            continue
        if re.match(r"^#{1,6}\s", s):
            continue
        if _line_looks_like_prose_line(s):
            continue
        if len(s) <= 120:
            return _normalize_cot_command(s)

    return "help"


class LLMCoTAgent(tales.Agent):

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
        self.cot_max_tokens = kwargs["cot_max_tokens"]

    @property
    def uid(self):
        return (
            f"LLMCoTAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_ct{self.cot_max_tokens}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "zero-shot-cot",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
            "cot_max_tokens": self.cot_max_tokens,
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

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": self.cot_max_tokens,
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
            "claude-3.7-sonnet",
        ]:
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            llm_kwargs.pop("seed")
            llm_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")

        response = self._llm_call_from_messages(messages, **llm_kwargs)

        raw_text = response.text()
        action = _parse_action_from_cot_response(raw_text)
        self.history.append((f"{obs}\n> ", f"{action}\n"))

        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": raw_text,
            "nb_tokens_prompt": self.token_counter(messages=messages),
            "nb_tokens_response": self.token_counter(text=raw_text),
        }

        stats["nb_tokens"] = stats["nb_tokens_prompt"] + stats["nb_tokens_response"]

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT_COT}]
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
            messages[1]["content"] = f"{SYSTEM_PROMPT_COT}\n\n{messages[1]['content']}"

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMCoTAgent settings")

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
        "--cot-max-tokens",
        type=int,
        default=512,
        help="Max tokens for the CoT reply (reasoning + ACTION line). Default: %(default)s",
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
    name="zero-shot-cot",
    desc=(
        "Zero-shot agent with chain-of-thought instructions; model reasons then outputs "
        "ACTION: <command>. Parsed command is what the environment receives."
    ),
    klass=LLMCoTAgent,
    add_arguments=build_argparser,
)
