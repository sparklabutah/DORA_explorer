import argparse

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


class ScheduledTempLLMAgent(tales.Agent):
    """
    Zero-shot LLM agent with an exponential temperature schedule: high early (exploration),
    decaying toward temp_end (exploitation), same growth curve as lambda_explore's lambda schedule.
    """

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

        self.conversation = kwargs["conversation"]

        self.horizon = max(1, kwargs["horizon"])
        self.temp_start = kwargs["temp_start"]
        self.temp_end = kwargs["temp_end"]
        self.temp_k = kwargs["temp_k"]
        self.t = 0

    def _current_temperature(self):
        frac = min(self.t / self.horizon, 1.0)
        k = self.temp_k
        growth = (np.exp(k * frac) - 1.0) / (np.exp(k) - 1.0)
        return self.temp_start + growth * (self.temp_end - self.temp_start)

    @property
    def uid(self):
        return (
            f"ScheduledTempLLMAgent_{self.llm}"
            f"_ts{self.temp_start}_te{self.temp_end}_k{self.temp_k}_h{self.horizon}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "scheduled-temp",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "conversation": self.conversation,
            "horizon": self.horizon,
            "temp_start": self.temp_start,
            "temp_end": self.temp_end,
            "temp_k": self.temp_k,
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
        temperature = float(self._current_temperature())

        llm_kwargs = {
            "temperature": temperature,
            "max_tokens": 100,
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

        action = response.text().strip()
        self.history.append((f"{obs}\n> ", f"{action}\n"))
        self.t += 1

        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.text(),
            "nb_tokens_prompt": self.token_counter(messages=messages),
            "nb_tokens_response": self.token_counter(text=response.text()),
        }

        stats["nb_tokens"] = stats["nb_tokens_prompt"] + stats["nb_tokens_response"]
        stats["policy"] = {
            "temperature": temperature,
            "t": self.t,
            "horizon": self.horizon,
        }

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, prev_action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": prev_action})

        messages.append({"role": "user", "content": observation})

        messages = merge_messages(messages)

        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("ScheduledTempLLMAgent settings")

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
    group.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Steps over which temperature moves from temp-start to temp-end. Default: %(default)s",
    )
    group.add_argument(
        "--temp-start",
        type=float,
        default=4.0,
        help="Temperature at step 0 (exploration). Default: %(default)s",
    )
    group.add_argument(
        "--temp-end",
        type=float,
        default=0.0,
        help="Temperature after horizon (exploitation). Default: %(default)s",
    )
    group.add_argument(
        "--temp-k",
        type=float,
        default=5.0,
        help="Exponential schedule sharpness (same role as lambda-k in lambda-explore). Default: %(default)s",
    )
    return parser


register(
    name="scheduled-temp",
    desc=(
        "Zero-shot LLM with exponential temperature schedule from temp-start to temp-end"
        " over a horizon (high T early, low T later)."
    ),
    klass=ScheduledTempLLMAgent,
    add_arguments=build_argparser,
)
