import argparse
import json
import os
import re
import time

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

LAMBDA_MIN_DEFAULT = 0.0
LAMBDA_MAX_DEFAULT = 5.0
EXPLORE_NUM_CANDIDATES = 20

TALES_LAMBDA_MIN_ENV = "TALES_LAMBDA_MIN"
TALES_LAMBDA_MAX_ENV = "TALES_LAMBDA_MAX"


def _float_from_env(var_name: str, default: float) -> float:
    raw = os.environ.get(var_name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def format_lambda_decision_prompt(lambda_min: float, lambda_max: float) -> str:
    return (
        "You chose to EXPLORE. Now choose the exploration parameter lambda.\n"
        "Lambda is a float. Lower values favor more exploratory sampling over candidate "
        "scores; higher values favor candidates that already score well.\n"
        f"The allowed range for lambda is [{lambda_min:.6g}, {lambda_max:.6g}] "
        "(inclusive).\n"
        "Return ONLY strict JSON on a single line with this shape:\n"
        '{"lambda":<float>}\n'
        "Rules:\n"
        f"- Choose lambda within [{lambda_min:.6g}, {lambda_max:.6g}].\n"
        f'- Example: {{"lambda": {(lambda_min + lambda_max) / 2:.4f}}}\n'
        "- Do not include markdown, code fences, or explanation."
    )


SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)

CANDIDATE_GENERATION_PROMPT = (
    "Based on the current game state, list {n} possible actions you could take."
    " Write one action per line as a short command phrase ."
    " Do not number the lines. Do not include any explanation."
)

MODE_DECISION_PROMPT = (
    "Decide whether to act GREEDY or EXPLORE."
    " You go GREEDY if you know what is your goal and how to achieve it. You are confident about your next action."
    " Choose EXPLORE only if you are stuck in an observation, action loop, or you are uncertain about your next action."
    " Return ONLY strict JSON on a single line with one of these shapes:\n"
    '{"mode":"GREEDY"}\n'
    '{"mode":"EXPLORE"}\n'
      "Rules:\n"
    '- mode must be exactly "GREEDY" or "EXPLORE".\n'
    "- Do not include markdown, code fences, or explanation."
)

class LambdaAutonomousAgent(tales.Agent):

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

        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        # Keep exploratory branching aligned with the requested design:
        # generate exactly 20 candidate commands when mode is EXPLORE.
        self.num_candidates = EXPLORE_NUM_CANDIDATES
        self.num_candidates_requested = kwargs["num_candidates"]
        self.gen_temp = kwargs["gen_temp"]
        self.max_action_space = kwargs["max_action_space"]
        self.micro_batch_size = kwargs["micro_batch_size"]
        # Keeping full per-token log-prob traces for every candidate is expensive
        # (GPU->CPU transfer + large JSON). Keep it optional.
        self.store_token_log_probs = kwargs.get("store_token_log_probs", False)

        self.lambda_min = _float_from_env(
            TALES_LAMBDA_MIN_ENV, float(kwargs["lambda_min"])
        )
        self.lambda_max = _float_from_env(
            TALES_LAMBDA_MAX_ENV, float(kwargs["lambda_max"])
        )
        if self.lambda_min > self.lambda_max:
            raise ValueError(
                f"lambda_min ({self.lambda_min}) must be <= lambda_max ({self.lambda_max}); "
                f"check --lambda-min/--lambda-max and "
                f"{TALES_LAMBDA_MIN_ENV}/{TALES_LAMBDA_MAX_ENV}."
            )
        self.decision_temp = kwargs["decision_temp"]
        self.decision_max_tokens = kwargs["decision_max_tokens"]

        self.seen_pairs = {}
        self.obs_tried = {}
        self.explore_path_uses = 0

        self.scoring_model_id = (
            kwargs["scoring_model"]
            or getattr(self.model, "model_name", None)
            or self.model.model_id
        )
        requested_data_parallel = kwargs.get("scoring_data_parallel", False)
        self.scoring_multi_gpu_available = (
            torch.cuda.is_available() and torch.cuda.device_count() > 1
        )
        # Keep the explicit CLI flag, but auto-enable DP when multiple GPUs are visible
        # so runs do not silently miss available parallelism.
        self.scoring_data_parallel_auto_enabled = (
            self.scoring_multi_gpu_available and not requested_data_parallel
        )
        self.scoring_data_parallel = (
            requested_data_parallel or self.scoring_data_parallel_auto_enabled
        )
        self._init_scorer(kwargs["scoring_dtype"])
        self._last_scoring_breakdown_seconds = {}

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
            # Full model on cuda:0, then DataParallel splits each forward batch across GPUs.
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

    def _sync_scoring_cuda(self):
        if not torch.cuda.is_available():
            return
        if self.scoring_data_parallel_active and torch.cuda.device_count() > 1:
            for idx in range(torch.cuda.device_count()):
                torch.cuda.synchronize(idx)
            return
        device = self._scoring_device()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

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
            f"LambdaAutonomousAgent_{self.llm}"
            f"_score{self.scoring_model_id}"
            f"_a{self.alpha}_b{self.beta}"
            f"_nc{self.num_candidates}_gt{self.gen_temp}"
            f"_lm{self.lambda_min}-{self.lambda_max}"
            f"_dt{self.decision_temp}"
            f"_m{self.max_action_space}"
            f"_mb{self.micro_batch_size}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_conv{self.conversation}"
        ).replace("/", "-")

    @property
    def params(self):
        return {
            "agent_type": "dora-auto-explore",
            "llm": self.llm,
            "scoring_model": self.scoring_model_id,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "conversation": self.conversation,
            "act_temp": self.act_temp,
            "alpha": self.alpha,
            "beta": self.beta,
            "num_candidates": self.num_candidates,
            "gen_temp": self.gen_temp,
            "max_action_space": self.max_action_space,
            "micro_batch_size": self.micro_batch_size,
            "store_token_log_probs": self.store_token_log_probs,
            "scoring_multi_gpu_available": self.scoring_multi_gpu_available,
            "scoring_data_parallel_auto_enabled": self.scoring_data_parallel_auto_enabled,
            "scoring_data_parallel_active": self.scoring_data_parallel_active,
            "scoring_data_parallel": self.scoring_data_parallel,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "decision_temp": self.decision_temp,
            "decision_max_tokens": self.decision_max_tokens,
        }

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

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

    def _model_kwargs(self, temperature, max_tokens):
        kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
            "claude-3.7-sonnet",
        ]:
            kwargs.pop("seed")
        if "gemini" in self.llm or "gemma" in self.llm:
            kwargs.pop("seed")
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        return kwargs

    def _greedy_action(self, messages):
        response = self._llm_call_from_messages(
            messages, **self._model_kwargs(0.0, max_tokens=100)
        )
        return response.text().strip(), response.text()

    # ------------------------------------------------------------------
    # Candidate generation & filtering
    # ------------------------------------------------------------------

    def _generate_candidates(self, messages):
        gen_messages = [msg.copy() for msg in messages]
        suffix = "\n\n" + CANDIDATE_GENERATION_PROMPT.format(n=self.num_candidates)
        gen_messages[-1]["content"] = gen_messages[-1]["content"] + suffix

        response = self._llm_call_from_messages(
            gen_messages, **self._model_kwargs(self.gen_temp, max_tokens=500)
        )
        raw_lines = response.text().strip().splitlines()
        candidates = []
        for line in raw_lines:
            cleaned = line.strip().lstrip("0123456789.-) ").strip()
            if cleaned:
                candidates.append(cleaned)
        return candidates

    def _is_valid_command_format(self, command: str):
        cmd = command.strip()
        if not cmd:
            return False
        if "\n" in cmd or "\r" in cmd:
            return False
        if len(cmd) > 120:
            return False

        # Common non-command patterns we do not want to execute.
        if cmd.startswith(("```", "#", ">", "* ", "- ")):
            return False

        return (
            re.match(r"^[A-Za-z0-9][A-Za-z0-9 '\-_,.?/():]*$", cmd) is not None
        )

    def _filter_candidates(self, candidates, admissible):
        seen = set()
        unique = []
        for c in candidates:
            key = c.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)

        format_valid = [c for c in unique if self._is_valid_command_format(c)]
        admissible_applied = bool(admissible)

        if admissible_applied:
            admissible_lower = {a.lower().strip() for a in admissible}
            admissible_valid = [
                c for c in format_valid if c.lower().strip() in admissible_lower
            ]
        else:
            admissible_valid = format_valid

        final = admissible_valid
        if len(final) > self.max_action_space:
            indices = self.rng.choice(len(final), size=self.max_action_space, replace=False)
            final = [final[i] for i in sorted(indices)]

        counts = {
            "raw_generated_count": len(candidates),
            "dedup_count": len(unique),
            "format_valid_count": len(format_valid),
            "admissible_filter_applied": admissible_applied,
            "admissible_valid_count": len(admissible_valid),
            "final_filtered_count": len(final),
        }
        return final, counts

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _messages_to_prompt(self, messages):
        lines = []
        for msg in messages:
            role = msg["role"].upper()
            lines.append(f"[{role}]\n{msg['content'].rstrip()}\n")
        lines.append("[ASSISTANT]\n")
        return "\n".join(lines)

    def _score_actions(self, prompt: str, actions):
        total_t0 = time.perf_counter()
        breakdown = {
            "prompt_tokenize_and_transfer": 0.0,
            "prompt_forward_cache_build": 0.0,
            "candidate_tokenize_and_transfer": 0.0,
            "candidate_continuation_forwards": 0.0,
            "postprocessing_other": 0.0,
        }
        results = []
        tokenizer = self.scoring_tokenizer
        model = self.scoring_model
        device = self._scoring_device()
        include_token_log_probs = self.store_token_log_probs

        t0 = time.perf_counter()
        prompt_enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192
        ).to(device)
        breakdown["prompt_tokenize_and_transfer"] += time.perf_counter() - t0
        prompt_ids = prompt_enc["input_ids"]
        prompt_attention_mask = prompt_enc["attention_mask"]
        prompt_len = prompt_ids.shape[1]
        max_response_tokens = max(0, 8192 - prompt_len)

        self._sync_scoring_cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            prompt_outputs = model(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                use_cache=True,
            )
        self._sync_scoring_cuda()
        breakdown["prompt_forward_cache_build"] += time.perf_counter() - t0
        prompt_past_key_values = prompt_outputs.past_key_values
        # DataParallel scatter works reliably with legacy tuple caches (tensor leaves).
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

            t0 = time.perf_counter()
            enc = tokenizer(
                batch_actions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_response_tokens,
                add_special_tokens=False,
            ).to(device)
            breakdown["candidate_tokenize_and_transfer"] += time.perf_counter() - t0

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
                self._sync_scoring_cuda()
                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs = model(
                        input_ids=continuation_input_ids,
                        attention_mask=model_attention_mask,
                        past_key_values=repeated_past_key_values,
                        # Keep cache mode enabled so Qwen attention path honors provided prefix cache.
                        use_cache=True,
                    )
                self._sync_scoring_cuda()
                breakdown["candidate_continuation_forwards"] += time.perf_counter() - t0
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
                token_log_probs_list = (
                    token_log_probs.detach().cpu().tolist() if include_token_log_probs else []
                )

                raw_stats.append(
                    {
                        "action": action_text,
                        "token_log_probs": token_log_probs_list,
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

        total_elapsed = time.perf_counter() - total_t0
        measured = (
            breakdown["prompt_tokenize_and_transfer"]
            + breakdown["prompt_forward_cache_build"]
            + breakdown["candidate_tokenize_and_transfer"]
            + breakdown["candidate_continuation_forwards"]
        )
        breakdown["postprocessing_other"] = max(0.0, total_elapsed - measured)
        breakdown["hf_scoring_total"] = total_elapsed
        self._last_scoring_breakdown_seconds = breakdown

        return results

    # ------------------------------------------------------------------
    # Decision parser
    # ------------------------------------------------------------------

    def _extract_json_object(self, text):
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _decide_mode(self, messages):
        mode_messages = [msg.copy() for msg in messages]
        mode_messages[-1]["content"] = (
            mode_messages[-1]["content"] + "\n\n" + MODE_DECISION_PROMPT
        )
        response = self._llm_call_from_messages(
            mode_messages,
            **self._model_kwargs(self.decision_temp, max_tokens=self.decision_max_tokens),
        )
        raw = response.text().strip()
        parsed = self._extract_json_object(raw)
        decision = {
            "mode_decision_raw": raw,
            "mode_decision_valid": False,
            "mode_requested": None,
            "mode_used": "GREEDY",
        }

        if not isinstance(parsed, dict):
            return decision

        mode = parsed.get("mode")
        decision["mode_requested"] = mode
        if not isinstance(mode, str):
            return decision

        mode = mode.strip().upper()
        if mode not in {"GREEDY", "EXPLORE"}:
            return decision

        decision["mode_used"] = mode
        decision["mode_decision_valid"] = True
        return decision

    def _decide_lambda(self, messages):
        lambda_messages = [msg.copy() for msg in messages]
        lambda_messages[-1]["content"] = (
            lambda_messages[-1]["content"]
            + "\n\n"
            + format_lambda_decision_prompt(self.lambda_min, self.lambda_max)
        )
        response = self._llm_call_from_messages(
            lambda_messages,
            **self._model_kwargs(self.decision_temp, max_tokens=self.decision_max_tokens),
        )
        raw = response.text().strip()
        parsed = self._extract_json_object(raw)

        decision = {
            "lambda_decision_raw": raw,
            "lambda_decision_valid": False,
            "lambda_requested": None,
            "lambda_used": None,
        }
        if not isinstance(parsed, dict):
            return decision

        lambda_requested = parsed.get("lambda")
        decision["lambda_requested"] = lambda_requested
        try:
            lambda_value = float(lambda_requested)
        except (TypeError, ValueError):
            return decision

        lambda_value = max(self.lambda_min, min(self.lambda_max, lambda_value))
        decision["lambda_used"] = lambda_value
        decision["lambda_decision_valid"] = True
        return decision

    # ------------------------------------------------------------------
    # Exploration path
    # ------------------------------------------------------------------

    def _explore(self, messages, obs_hash, lambda_value, admissible):
        raw_candidates = self._generate_candidates(messages)
        filtered, filter_counts = self._filter_candidates(raw_candidates, admissible)
        tried = self.obs_tried.get(obs_hash, set())
        novel = [c for c in filtered if c.lower().strip() not in tried]
        # If every filtered candidate was already tried at this observation, do not
        # re-score repeats — fall back to greedy (caller handles None).
        if len(novel) == 0:
            explore = {
                "raw_candidates": raw_candidates,
                "filtered_candidates": filtered,
                "novel_candidates": novel,
                "candidates_scored": [],
                "scored_actions": None,
                "selection_probs": None,
                "explore_fallback_reason": "no_novel_candidates",
                "raw_generated_count": filter_counts["raw_generated_count"],
                "dedup_count": filter_counts["dedup_count"],
                "format_valid_count": filter_counts["format_valid_count"],
                "admissible_filter_applied": filter_counts["admissible_filter_applied"],
                "admissible_valid_count": filter_counts["admissible_valid_count"],
                "final_filtered_count": filter_counts["final_filtered_count"],
            }
            return None, explore

        explore = {
            "raw_candidates": raw_candidates,
            "filtered_candidates": filtered,
            "novel_candidates": novel,
            "candidates_scored": [],
            "scored_actions": None,
            "selection_probs": None,
            "raw_generated_count": filter_counts["raw_generated_count"],
            "dedup_count": filter_counts["dedup_count"],
            "format_valid_count": filter_counts["format_valid_count"],
            "admissible_filter_applied": filter_counts["admissible_filter_applied"],
            "admissible_valid_count": filter_counts["admissible_valid_count"],
            "final_filtered_count": filter_counts["final_filtered_count"],
        }

        # Exactly one untried action at this observation: take it (no softmax / scoring).
        if len(novel) == 1:
            action = novel[0]
            explore["candidates_scored"] = [action]
            explore["selection_probs"] = [1.0]
            explore["explore_single_novel"] = True
            return action, explore

        candidates_to_score = novel
        explore["candidates_scored"] = candidates_to_score

        prompt = self._messages_to_prompt(messages)
        scored = self._score_actions(prompt, candidates_to_score)
        scores = np.array([row["final_score"] for row in scored], dtype=np.float64)
        scaled = lambda_value * scores
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= probs.sum()
        idx = int(self.rng.choice(len(candidates_to_score), p=probs))
        action = candidates_to_score[idx]

        explore["scored_actions"] = scored
        explore["selection_probs"] = probs.tolist()
        return action, explore

    # ------------------------------------------------------------------
    # Main act loop
    # ------------------------------------------------------------------

    def act(self, obs, reward, done, infos):
        if done:
            self.seen_pairs.clear()
            self.obs_tried.clear()
            self.explore_path_uses = 0

        messages = self.build_messages(f"{obs}\n> ")
        obs_hash = hash(obs.strip())
        admissible = infos.get("admissible_commands") or []

        mode_decision = self._decide_mode(messages)
        lambda_decision = {
            "lambda_decision_raw": None,
            "lambda_decision_valid": False,
            "lambda_requested": None,
            "lambda_used": None,
        }

        action = None
        response_text = None
        explore = {
            "raw_candidates": [],
            "filtered_candidates": [],
            "novel_candidates": [],
            "candidates_scored": [],
            "scored_actions": None,
            "selection_probs": None,
            "raw_generated_count": 0,
            "dedup_count": 0,
            "format_valid_count": 0,
            "admissible_filter_applied": False,
            "admissible_valid_count": 0,
            "final_filtered_count": 0,
        }
        mode_used = mode_decision["mode_used"]

        if mode_used == "GREEDY":
            action, response_text = self._greedy_action(messages)
        else:
            lambda_decision = self._decide_lambda(messages)
            if not lambda_decision["lambda_decision_valid"]:
                mode_used = "GREEDY_FALLBACK"
                action, response_text = self._greedy_action(messages)
            else:
                action, explore = self._explore(
                    messages,
                    obs_hash=obs_hash,
                    lambda_value=lambda_decision["lambda_used"],
                    admissible=admissible,
                )
                if action is None:
                    mode_used = "GREEDY_FALLBACK"
                    action, response_text = self._greedy_action(messages)
                else:
                    self.explore_path_uses += 1
                    if explore.get("explore_single_novel"):
                        response_text = (
                            f"AUTONOMOUS-EXPLORE: single_novel (no softmax),"
                            f" selected={action}"
                        )
                    else:
                        response_text = (
                            f"AUTONOMOUS-EXPLORE: lambda={lambda_decision['lambda_used']:.4f},"
                            f" selected={action},"
                            f" candidates={len(explore['candidates_scored'])}"
                        )

        final_key = action.lower().strip()
        pair = (obs_hash, final_key)
        loop_count = self.seen_pairs.get(pair, 0)
        self.seen_pairs[pair] = loop_count + 1

        if obs_hash not in self.obs_tried:
            self.obs_tried[obs_hash] = set()
        self.obs_tried[obs_hash].add(final_key)

        self.history.append((f"{obs}\n> ", f"{action}\n"))

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
        mode_requested = mode_decision["mode_requested"]
        decision_valid = mode_decision["mode_decision_valid"] and (
            mode_requested == "GREEDY" or lambda_decision["lambda_decision_valid"]
        )
        stats["policy"] = {
            "decision_valid": decision_valid,
            "decision_raw": mode_decision["mode_decision_raw"],
            "mode_decision_raw": mode_decision["mode_decision_raw"],
            "lambda_decision_raw": lambda_decision["lambda_decision_raw"],
            "mode_decision_valid": mode_decision["mode_decision_valid"],
            "lambda_decision_valid": lambda_decision["lambda_decision_valid"],
            "mode_requested": mode_requested,
            "mode_used": mode_used,
            "lambda_requested": lambda_decision["lambda_requested"],
            "lambda_used": lambda_decision["lambda_used"],
            "explore_path_uses": self.explore_path_uses,
            "num_candidates_requested": self.num_candidates_requested,
            "num_candidates_used": self.num_candidates,
            "loop_count_for_final_pair": loop_count + 1,
            "obs_tried_count": len(self.obs_tried.get(obs_hash, set())),
            "raw_candidates_count": len(explore["raw_candidates"]),
            "raw_generated_count": explore["raw_generated_count"],
            "dedup_count": explore["dedup_count"],
            "format_valid_count": explore["format_valid_count"],
            "admissible_filter_applied": explore["admissible_filter_applied"],
            "admissible_valid_count": explore["admissible_valid_count"],
            "filtered_candidates_count": len(explore["filtered_candidates"]),
            "final_filtered_count": explore["final_filtered_count"],
            "novel_candidates_count": len(explore["novel_candidates"]),
            "candidates_scored_count": len(explore["candidates_scored"]),
            "scored_actions": explore["scored_actions"],
            "selection_probs": explore["selection_probs"],
            "explore_single_novel": explore.get("explore_single_novel", False),
            "explore_fallback_reason": explore.get("explore_fallback_reason"),
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
    group = parser.add_argument_group("LambdaAutonomousAgent settings")
    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM alias used for action generation and autonomous mode/lambda decision.",
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
        help="Temperature for greedy action LLM call. Default: %(default)s",
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
        "--decision-temp",
        type=float,
        default=0.0,
        help="Temperature for autonomous mode/lambda JSON decision call. Default: %(default)s",
    )
    group.add_argument(
        "--decision-max-tokens",
        type=int,
        default=64,
        help="Max tokens for autonomous mode/lambda JSON decision call. Default: %(default)s",
    )
    group.add_argument(
        "--lambda-min",
        type=float,
        default=LAMBDA_MIN_DEFAULT,
        help=(
            "Lower bound for model-chosen exploration lambda. Default: %(default)s. "
            f"If set, environment variable {TALES_LAMBDA_MIN_ENV} overrides this value."
        ),
    )
    group.add_argument(
        "--lambda-max",
        type=float,
        default=LAMBDA_MAX_DEFAULT,
        help=(
            "Upper bound for model-chosen exploration lambda. Default: %(default)s. "
            f"If set, environment variable {TALES_LAMBDA_MAX_ENV} overrides this value."
        ),
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
        "--num-candidates",
        type=int,
        default=20,
        help="Candidate actions to generate during exploration. Default: %(default)s",
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
        "--store-token-log-probs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store full per-token log-prob arrays in scored_actions. "
            "This increases scoring latency and log size. Default: %(default)s"
        ),
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
    name="dora-auto-explore",
    desc=(
        "Autonomous controller agent: model emits strict JSON each step to decide"
        " GREEDY vs EXPLORE mode. In explore mode, model also chooses bounded"
        " lambda for candidate-score sampling."
    ),
    klass=LambdaAutonomousAgent,
    add_arguments=build_argparser,
)
