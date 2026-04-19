import argparse
import os

import llm
import numpy as np
import torch
import torch.nn.functional as F
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

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

CANDIDATE_GENERATION_PROMPT = (
    "Based on the current game state, list {n} possible actions you could take."
    " Write one action per line as a short command phrase (e.g., 'get lamp', 'go north')."
    " Do not number the lines. Do not include any explanation."
)


class LambdaExploreAgent(tales.Agent):
    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # Provide API key if needed for generation fallback.
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
        self.act_temp = kwargs["act_temp"]

        # Lambda policy knobs.
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.lambda_start = kwargs["lambda_start"]
        self.lambda_end = kwargs["lambda_end"]
        self.lambda_k = kwargs["lambda_k"]
        self.horizon = max(1, kwargs["horizon"])
        self.max_action_space = kwargs["max_action_space"]
        self.micro_batch_size = kwargs["micro_batch_size"]
        self.num_candidates = kwargs["num_candidates"]
        self.gen_temp = kwargs["gen_temp"]
        self.t = 0

        # Scoring model for token-level log-prob/variance.
        self.scoring_model_id = (
            kwargs["scoring_model"]
            or getattr(self.model, "model_name", None)
            or self.model.model_id
        )
        requested_data_parallel = kwargs.get("scoring_data_parallel", False)
        self.scoring_multi_gpu_available = (
            torch.cuda.is_available() and torch.cuda.device_count() > 1
        )
        self.scoring_data_parallel_auto_enabled = (
            self.scoring_multi_gpu_available and not requested_data_parallel
        )
        self.scoring_data_parallel = (
            requested_data_parallel or self.scoring_data_parallel_auto_enabled
        )
        self._init_scorer(kwargs["scoring_dtype"])

    def _init_scorer(self, dtype_name: str):
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(
            self.scoring_model_id, token=hf_token
        )
        if self.scoring_tokenizer.pad_token_id is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype_name]
        use_dp = self.scoring_data_parallel and self.scoring_multi_gpu_available
        self.scoring_data_parallel_active = use_dp
        if use_dp:
            self.scoring_model = AutoModelForCausalLM.from_pretrained(
                self.scoring_model_id,
                token=hf_token,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            self.scoring_model = self.scoring_model.to(torch.device("cuda:0"))
            self.scoring_model = torch.nn.DataParallel(self.scoring_model)
        else:
            self.scoring_model = AutoModelForCausalLM.from_pretrained(
                self.scoring_model_id,
                token=hf_token,
                torch_dtype=torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        self.scoring_model.eval()

    def _scoring_device(self):
        return next(self.scoring_model.parameters()).device

    def _repeat_past_key_values(self, past_key_values, batch_size: int):
        """Broadcast prompt cache from batch=1 to batch=batch_size."""
        if batch_size == 1:
            return past_key_values

        if hasattr(past_key_values, "batch_repeat_interleave"):
            return past_key_values.batch_repeat_interleave(batch_size)

        if (
            hasattr(past_key_values, "to_legacy_cache")
            and hasattr(type(past_key_values), "from_legacy_cache")
        ):
            legacy = past_key_values.to_legacy_cache()
            repeated_legacy = self._repeat_past_key_values(legacy, batch_size)
            return type(past_key_values).from_legacy_cache(repeated_legacy)

        if isinstance(past_key_values, torch.Tensor):
            if past_key_values.shape[0] != 1:
                raise ValueError(
                    f"Cannot broadcast cache tensor with batch={past_key_values.shape[0]}"
                )
            repeat_dims = [batch_size] + [1] * (past_key_values.dim() - 1)
            return past_key_values.repeat(*repeat_dims)

        if isinstance(past_key_values, tuple):
            return tuple(
                self._repeat_past_key_values(item, batch_size)
                for item in past_key_values
            )

        if isinstance(past_key_values, list):
            return [
                self._repeat_past_key_values(item, batch_size)
                for item in past_key_values
            ]

        raise TypeError(
            f"Unsupported past_key_values type for broadcast: {type(past_key_values)}"
        )

    @property
    def uid(self):
        return (
            f"LambdaExploreAgent_{self.llm}"
            f"_score{self.scoring_model_id}"
            f"_a{self.alpha}_b{self.beta}"
            f"_l{self.lambda_start}-{self.lambda_end}"
            f"_h{self.horizon}"
            f"_m{self.max_action_space}"
            f"_nc{self.num_candidates}_gt{self.gen_temp}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_conv{self.conversation}"
        ).replace("/", "-")

    @property
    def params(self):
        return {
            "agent_type": "lambda-explore",
            "llm": self.llm,
            "scoring_model": self.scoring_model_id,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "conversation": self.conversation,
            "act_temp": self.act_temp,
            "alpha": self.alpha,
            "beta": self.beta,
            "lambda_start": self.lambda_start,
            "lambda_end": self.lambda_end,
            "lambda_k": self.lambda_k,
            "horizon": self.horizon,
            "max_action_space": self.max_action_space,
            "micro_batch_size": self.micro_batch_size,
            "num_candidates": self.num_candidates,
            "gen_temp": self.gen_temp,
            "scoring_multi_gpu_available": self.scoring_multi_gpu_available,
            "scoring_data_parallel_auto_enabled": self.scoring_data_parallel_auto_enabled,
            "scoring_data_parallel_active": self.scoring_data_parallel_active,
            "scoring_data_parallel": self.scoring_data_parallel,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()  # Forces response materialization.
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None
        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def _fallback_generate_action(self, messages):
        llm_kwargs = {
            "temperature": self.act_temp,
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
        return action, response.text()

    def _generate_candidates(self, messages):
        gen_messages = [msg.copy() for msg in messages]
        suffix = "\n\n" + CANDIDATE_GENERATION_PROMPT.format(n=self.num_candidates)
        gen_messages[-1]["content"] = gen_messages[-1]["content"] + suffix

        llm_kwargs = {
            "temperature": self.gen_temp,
            "max_tokens": 500,
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

        response = self._llm_call_from_messages(gen_messages, **llm_kwargs)
        raw_lines = response.text().strip().splitlines()
        candidates = []
        for line in raw_lines:
            cleaned = line.strip().lstrip("0123456789.-) ").strip()
            if cleaned:
                candidates.append(cleaned)
        return candidates

    def _filter_candidates(self, candidates, admissible):
        seen = set()
        unique = []
        for c in candidates:
            key = c.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)

        if admissible:
            admissible_lower = {a.lower().strip() for a in admissible}
            unique = [c for c in unique if c.lower().strip() in admissible_lower]

        if len(unique) > self.max_action_space:
            indices = self.rng.choice(len(unique), size=self.max_action_space, replace=False)
            unique = [unique[i] for i in sorted(indices)]

        return unique

    def _current_lambda(self):

        frac = min(self.t / self.horizon, 1.0)
        growth = (np.exp(self.lambda_k * frac) - 1.0) / (np.exp(self.lambda_k) - 1.0)
        return self.lambda_start + growth * (self.lambda_end - self.lambda_start)

    def _messages_to_prompt(self, messages):
        lines = []
        for msg in messages:
            role = msg["role"].upper()
            lines.append(f"[{role}]\n{msg['content'].rstrip()}\n")
        lines.append("[ASSISTANT]\n")
        return "\n".join(lines)

    def _score_actions(self, prompt: str, actions):
        results = []
        tokenizer = self.scoring_tokenizer
        model = self.scoring_model
        device = self._scoring_device()

        prompt_enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192
        ).to(device)
        prompt_ids = prompt_enc["input_ids"]
        prompt_attention_mask = prompt_enc["attention_mask"]
        prompt_len = prompt_ids.shape[1]
        max_response_tokens = max(0, 8192 - prompt_len)

        with torch.no_grad():
            prompt_outputs = model(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                use_cache=True,
            )
        prompt_past_key_values = prompt_outputs.past_key_values
        if hasattr(prompt_past_key_values, "to_legacy_cache"):
            prompt_past_key_values = prompt_past_key_values.to_legacy_cache()
        prompt_last_logits = prompt_outputs.logits[:, -1, :]
        prompt_last_log_probs = F.log_softmax(prompt_last_logits, dim=-1)

        raw_stats = []
        for i in range(0, len(actions), self.micro_batch_size):
            batch_actions = actions[i : i + self.micro_batch_size]
            if max_response_tokens <= 0:
                for action_text in batch_actions:
                    raw_stats.append(
                        {
                            "action": action_text,
                            "token_log_probs": [],
                            "mean_log_prob": 0.0,
                            "variance": 0.0,
                        }
                    )
                continue

            enc = tokenizer(
                batch_actions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_response_tokens,
                add_special_tokens=False,
            ).to(device)

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)

            continuation_token_log_probs = None
            if seq_len > 1:
                continuation_input_ids = input_ids[:, :-1]
                continuation_attention_mask = attention_mask[:, :-1]
                batch_prompt_attention = prompt_attention_mask.expand(batch_size, -1)
                model_attention_mask = torch.cat(
                    [batch_prompt_attention, continuation_attention_mask], dim=1
                )
                repeated_past_key_values = self._repeat_past_key_values(
                    prompt_past_key_values, batch_size
                )
                with torch.no_grad():
                    outputs = model(
                        input_ids=continuation_input_ids,
                        attention_mask=model_attention_mask,
                        past_key_values=repeated_past_key_values,
                        use_cache=True,
                    )
                continuation_logits = outputs.logits
                continuation_labels = input_ids[:, 1:]
                continuation_token_log_probs = (
                    F.log_softmax(continuation_logits, dim=-1)
                    .gather(2, continuation_labels.unsqueeze(-1))
                    .squeeze(-1)
                )

            for b in range(batch_size):
                full_len = attention_mask[b].sum().item()
                resp_len = int(full_len)
                action_text = batch_actions[b]

                if resp_len <= 0:
                    raw_stats.append(
                        {
                            "action": action_text,
                            "token_log_probs": [],
                            "mean_log_prob": 0.0,
                            "variance": 0.0,
                        }
                    )
                    continue

                first_token_id = input_ids[b, 0].view(1, 1)
                first_token_log_prob = (
                    prompt_last_log_probs.gather(1, first_token_id).squeeze(1)
                )
                token_log_probs = first_token_log_prob
                if resp_len > 1 and continuation_token_log_probs is not None:
                    token_log_probs = torch.cat(
                        [first_token_log_prob, continuation_token_log_probs[b, : resp_len - 1]],
                        dim=0,
                    )

                raw_stats.append(
                    {
                        "action": action_text,
                        "token_log_probs": token_log_probs.detach().cpu().tolist(),
                        "mean_log_prob": float(token_log_probs.mean().item()),
                        "variance": float(token_log_probs.var(unbiased=False).item()),
                    }
                )

        means = [r["mean_log_prob"] for r in raw_stats]
        variances = [r["variance"] for r in raw_stats]
        min_mean, max_mean = min(means), max(means)
        min_var, max_var = min(variances), max(variances)
        eps = 1e-8

        for row in raw_stats:
            if max_mean == min_mean:
                norm_mean = 0.5
            else:
                norm_mean = (row["mean_log_prob"] - min_mean) / (max_mean - min_mean + eps)

            if max_var == min_var:
                norm_var = 0.5
            else:
                norm_var = (row["variance"] - min_var) / (max_var - min_var + eps)

            row["norm_mean_log_prob"] = norm_mean
            row["norm_variance"] = norm_var
            row["final_score"] = self.alpha * norm_mean - self.beta * norm_var
            results.append(row)

        return results

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        admissible = infos.get("admissible_commands") or []

        raw_candidates = self._generate_candidates(messages)
        filtered = self._filter_candidates(raw_candidates, admissible)
        used_lambda_policy = len(filtered) >= 2

        if used_lambda_policy:
            prompt = self._messages_to_prompt(messages)
            scored = self._score_actions(prompt, filtered)
            scores = np.array([row["final_score"] for row in scored], dtype=np.float64)
            lambda_val = self._current_lambda()
            scaled = lambda_val * scores
            scaled -= np.max(scaled)
            probs = np.exp(scaled)
            probs /= probs.sum()
            idx = int(self.rng.choice(len(filtered), p=probs))
            action = filtered[idx]
            response_text = (
                f"lambda={lambda_val:.4f}, selected={action},"
                f" candidates={len(raw_candidates)}, filtered={len(filtered)}"
            )
        else:
            scored = None
            probs = None
            lambda_val = self._current_lambda()
            action, response_text = self._fallback_generate_action(messages)

        self.history.append((f"{obs}\n> ", f"{action}\n"))
        self.t += 1

        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response_text,
            "nb_tokens_prompt": self.token_counter(messages=messages),
            "nb_tokens_response": self.token_counter(text=action),
            "nb_tokens_thinking": 0,
        }
        stats["nb_tokens"] = (
            stats["nb_tokens_prompt"]
            + stats["nb_tokens_response"]
            + stats["nb_tokens_thinking"]
        )
        stats["policy"] = {
            "used_lambda_policy": used_lambda_policy,
            "t": self.t,
            "lambda": lambda_val,
            "raw_candidates": raw_candidates,
            "filtered_candidates": filtered,
            "scored_actions": scored,
            "selection_probs": probs.tolist() if probs is not None else None,
        }

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                obs = f"// History has been truncated to the last {limit} steps.\n...\n> "
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


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LambdaExploreAgent settings")
    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM alias used for API fallback generation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for random sampling. Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for fallback generation path. Default: %(default)s",
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
        "--alpha",
        type=float,
        default=0.8,
        help="Weight for normalized mean token log-probability. Default: %(default)s",
    )
    group.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Weight for normalized log-probability variance penalty. Default: %(default)s",
    )
    group.add_argument(
        "--lambda-start",
        type=float,
        default=0.0,
        help="Initial lambda (exploration strength). Default: %(default)s",
    )
    group.add_argument(
        "--lambda-end",
        type=float,
        default=40.0,
        help="Final lambda (exploitation strength). Default: %(default)s",
    )
    group.add_argument(
        "--lambda-k",
        type=float,
        default=5.0,
        help="Exponential growth factor for lambda schedule. Default: %(default)s",
    )
    group.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Horizon used by lambda schedule. Default: %(default)s",
    )
    group.add_argument(
        "--num-candidates",
        type=int,
        default=20,
        help="Number of candidate actions the LLM is asked to generate. Default: %(default)s",
    )
    group.add_argument(
        "--gen-temp",
        type=float,
        default=0.7,
        help="Temperature for candidate generation LLM call. Default: %(default)s",
    )
    group.add_argument(
        "--max-action-space",
        type=int,
        default=20,
        help="Cap on candidates to score; excess are randomly sampled down. Default: %(default)s",
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size for scoring actions with HF model. Default: %(default)s",
    )
    group.add_argument(
        "--scoring-model",
        default=None,
        help="HF model id used for sequence scoring. Defaults to --llm model_id.",
    )
    group.add_argument(
        "--scoring-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for scoring model. Default: %(default)s",
    )
    group.add_argument(
        "--scoring-data-parallel",
        action="store_true",
        help=(
            "Request torch.nn.DataParallel for scorer. Multi-GPU runs also auto-enable this "
            "when available so parallelism is not silently missed."
        ),
    )
    return parser


register(
    name="lambda-explore",
    desc=(
        "Generative lambda exploration policy: prompts the LLM to generate candidate"
        " actions, filters duplicates/invalids, scores survivors using token"
        " log-probability and variance, then samples via an exponential lambda schedule."
    ),
    klass=LambdaExploreAgent,
    add_arguments=build_argparser,
)
