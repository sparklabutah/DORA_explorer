from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prompts import ARM_NAMES, candidate_generation_prompt
from score import score_responses_same_prompt

from .llm import parse_bandit_color_strict, query_llm

_ANSWER_BLOCK = re.compile(
    r"<answer[^>]*>.*?</answer>", flags=re.IGNORECASE | re.DOTALL
)


def _extract_candidates(raw: str) -> Tuple[List[str], List[str]]:
    """Pull every ``<Answer>…</Answer>`` block from the raw generation.

    Returns ``(valid, invalid)`` where *valid* contains the original answer
    blocks that include a recognisable color (exact-text duplicates removed)
    and *invalid* collects those that do not.
    """
    seen_text: set[str] = set()
    valid: List[str] = []
    invalid: List[str] = []

    for m in _ANSWER_BLOCK.finditer(raw):
        text = m.group(0).strip()
        if text in seen_text:
            continue
        seen_text.add(text)
        if parse_bandit_color_strict(text) is None:
            invalid.append(text)
        else:
            valid.append(text)

    return valid, invalid


class LambdaPolicyLLMAgent:
    """
    DORA on the bandit: generate → filter → score → λ-scheduled softmax sample.
    """

    def __init__(
        self,
        horizon: int = 200,
        alpha: float = 0.8,
        beta: float = 0.2,
        lambda_start: float = 0.0,
        lambda_end: float = 40.0,
        lambda_k: float = 5.0,
        micro_batch_size: int = 2,
        num_candidates: int = 10,
        gen_temp: float = 0.7,
        gen_max_new_tokens: int = 500,
        seed: Optional[int] = None,
    ):
        self.horizon = horizon
        self.alpha = alpha
        self.beta = beta
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.lambda_k = float(lambda_k)
        self.micro_batch_size = micro_batch_size
        self.num_candidates = int(num_candidates)
        self.gen_temp = float(gen_temp)
        self.gen_max_new_tokens = int(gen_max_new_tokens)
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def current_lambda(self) -> float:
        k = self.lambda_k
        frac = self.t / max(self.horizon, 1)
        growth = (np.exp(k * frac) - 1.0) / (np.exp(k) - 1.0)
        return float(self.lambda_start + growth * (self.lambda_end - self.lambda_start))

    def _greedy_fallback(
        self, system_prompt: str, user_prompt: str
    ) -> Tuple[int, str, Dict[str, Any]]:
        raw = query_llm(
            system_prompt,
            user_prompt,
            temperature=0.0,
            max_new_tokens=80,
        )
        idx = parse_bandit_color_strict(raw)
        if idx is None:
            idx = int(self.rng.integers(len(ARM_NAMES)))
        diag: Dict[str, Any] = {
            "path": "greedy_fallback",
            "raw_response": raw,
        }
        return idx, raw, diag

    def act(
        self, system_prompt: str, user_prompt: str
    ) -> Tuple[int, Dict[str, Any], np.ndarray]:
        scoring_prompt = system_prompt + "\n" + user_prompt
        gen_user = user_prompt + "\n\n" + candidate_generation_prompt(self.num_candidates)

        raw_gen = query_llm(
            system_prompt,
            gen_user,
            temperature=self.gen_temp,
            max_new_tokens=self.gen_max_new_tokens,
        )
        candidates, invalid_lines = _extract_candidates(raw_gen)

        invalid_info: Dict[str, Any] = {
            "num_invalid": len(invalid_lines),
            "invalid_lines": invalid_lines,
        }

        if len(candidates) >= 2:
            scored = score_responses_same_prompt(
                prompt=scoring_prompt,
                responses=candidates,
                micro_batch_size=self.micro_batch_size,
                alpha=self.alpha,
                beta=self.beta,
            )
            scores = np.array([r["final_score"] for r in scored], dtype=np.float64)
            lam = self.current_lambda()
            scaled = lam * scores
            scaled -= np.max(scaled)
            probs = np.exp(scaled)
            probs /= probs.sum()
            pick = int(self.rng.choice(len(candidates), p=probs))
            action_idx = parse_bandit_color_strict(candidates[pick])
            self.t += 1
            diagnostics: Dict[str, Any] = {
                "path": "dora_generate_score_sample",
                "raw_generation": raw_gen,
                "candidates": candidates,
                "selected": candidates[pick],
                "scored": scored,
                "lambda": lam,
                **invalid_info,
            }
            return action_idx, diagnostics, probs

        if len(candidates) == 1:
            action_idx = parse_bandit_color_strict(candidates[0])
            self.t += 1
            diagnostics = {
                "path": "single_candidate_no_softmax",
                "raw_generation": raw_gen,
                "candidates": candidates,
                **invalid_info,
            }
            return action_idx, diagnostics, np.array([1.0])

        action_idx, raw, diag = self._greedy_fallback(system_prompt, user_prompt)
        self.t += 1
        diagnostics = {
            "path": "greedy_fallback_no_valid_candidates",
            "raw_generation": raw_gen,
            **invalid_info,
            **diag,
        }
        return action_idx, diagnostics, np.array([1.0])
