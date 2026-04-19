"""Canonical ReAct agent (Yao et al., 2022 — https://arxiv.org/abs/2210.03629).

Each step the model sees the full Thought-Action-Observation trace and produces
a single response containing both ``Thought: ...`` and ``Action: ...``.  The
thought is persisted in history so the agent can reference its own prior
reasoning in future steps.
"""

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
from termcolor import colored

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    log,
    merge_messages,
    messages2conversation,
)

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score.\n\n"
    "At each step you receive an observation from the game. You must respond with a Thought "
    "and an Action, in that order.\n\n"
    "Format your response exactly as:\n"
    "Thought: <brief reasoning about the situation and what to do>\n"
    "Action: <single short game command, e.g. get lamp>\n\n"
    "Rules:\n"
    "- Always output exactly one Thought line followed by exactly one Action line.\n"
    "- The Action must be a bare game command (no backticks, no quotes).\n"
    " When stuck, try using the `help` command to see what commands are available."
)

_MAX_CMD_LEN = 200


def _strip_cmd_wrapping(s: str) -> str:
    s = s.strip()
    s = s.strip("`\"'")
    return s.strip()


def _normalize_command(cmd: str) -> str:
    """Single-line, bounded-length command for the game."""
    if not cmd:
        return "help"
    one = cmd.split("\n")[0].strip()
    one = re.sub(r"\s+", " ", one)
    one = _strip_cmd_wrapping(one)
    one = re.sub(r"[\s.]+$", "", one)
    if len(one) > _MAX_CMD_LEN:
        one = one[:_MAX_CMD_LEN].rsplit(" ", 1)[0] if " " in one else one[:_MAX_CMD_LEN]
    return one or "help"


def _parse_react_response(raw: str) -> tuple[str, str]:
    """Parse a ReAct response into (thought, action).

    Returns the full thought text and the extracted game command.
    """
    text = (raw or "").strip()
    if not text:
        return "", "help"

    # Extract thought (everything from "Thought:" up to "Action:")
    thought = ""
    thought_match = re.search(
        r"^Thought:\s*(.*?)(?=^Action:|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action — last "Action:" line wins
    action = ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        stripped = re.sub(r"^\*+\s*", "", stripped)
        m = re.match(r"^Action:\s*(.*)$", stripped, flags=re.IGNORECASE)
        if m and m.group(1).strip():
            action = _normalize_command(m.group(1))
            break

    if not action:
        # Fallback: inline Action: somewhere in a line
        for line in reversed(text.splitlines()):
            m = re.search(r"\bAction:\s*([^\n]+)", line.strip(), flags=re.IGNORECASE)
            if m:
                action = _normalize_command(m.group(1))
                if action:
                    break

    if not action:
        # Last resort: take the last short non-prose line
        for line in reversed(text.splitlines()):
            s = line.strip()
            if not s or s.startswith(("#", "```", "---", "**", "Thought:")):
                continue
            if len(s) <= 120 and len(s.split()) <= 12:
                action = _normalize_command(s)
                break

    return thought, action or "help"


class ReactCanonicalAgent(tales.Agent):

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

        self.history: list[tuple[str, str, str]] = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.max_tokens = kwargs["max_tokens"]
        self.conversation = kwargs["conversation"]

    @property
    def uid(self):
        return (
            f"ReactCanonicalAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_n{self.max_tokens}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "react-canonical",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "max_tokens": self.max_tokens,
            "conversation": self.conversation,
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
        messages = self.build_messages(obs)

        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": self.max_tokens,
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

        thought, action = _parse_react_response(raw_text)
        log.debug(colored(f"Thought: {thought}", "green"))
        log.debug(colored(f"Action: {action}", "cyan"))

        self.history.append((obs, thought, action))

        nb_tokens_prompt = self.token_counter(messages=messages)
        nb_tokens_response = self.token_counter(text=raw_text)
        nb_tokens_thinking = self.token_counter(text=thought) if thought else 0
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": raw_text,
            "thinking": thought,
            "nb_tokens_prompt": nb_tokens_prompt,
            "nb_tokens_response": nb_tokens_response,
            "nb_tokens_thinking": nb_tokens_thinking,
            "nb_tokens": nb_tokens_prompt + nb_tokens_response,
        }

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, thought, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                obs_content = (
                    f"[History truncated to the last {limit} steps.]\n"
                    f"...\n"
                    f"Observation: ..."
                )
            else:
                obs_content = f"Observation: {obs}"

            messages.append({"role": "user", "content": obs_content})

            assistant_text = f"Thought: {thought}\nAction: {action}"
            messages.append({"role": "assistant", "content": assistant_text})

        messages.append(
            {"role": "user", "content": f"Observation: {observation}"}
        )

        messages = merge_messages(messages)

        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"
            messages.pop(0)

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("ReactCanonicalAgent settings")

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
        help="Temperature for LLM generation (thought + action). Default: %(default)s",
    )
    group.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for the combined thought + action response. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="react-canonical",
    desc=(
        "Canonical ReAct agent (Yao et al., 2022). Generates interleaved "
        "Thought/Action traces in a single LLM call; the full reasoning "
        "trace is persisted across steps."
    ),
    klass=ReactCanonicalAgent,
    add_arguments=build_argparser,
)
