"""LLM-based bandit agent using HuggingFace Transformers."""

from typing import Optional, Tuple
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

from prompts import ARM_NAMES

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()

_COLOR_WORD = re.compile(r"\b(blue|green|red|yellow|purple)\b", flags=re.IGNORECASE)
_ANSWER_BLOCK = re.compile(
    r"<answer[^>]*>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL
)


def parse_bandit_color_strict(response: str) -> Optional[int]:
    """
    Valid iff some <Answer>...</Answer> block contains a canonical color.
    If multiple colors appear, use the last occurrence (final choice).
    No tail/random fallback.
    """
    if not response or not response.strip():
        return None
    text = response.lower()
    best_idx = None
    best_pos = -1
    for m in _ANSWER_BLOCK.finditer(text):
        inner = m.group(1)
        for cm in _COLOR_WORD.finditer(inner):
            w = cm.group(1).lower()
            if w in ARM_NAMES:
                p = m.start(1) + cm.start()
                if p >= best_pos:
                    best_pos = p
                    best_idx = ARM_NAMES.index(w)
    return best_idx


def parse_candidate_line(line: str) -> Optional[int]:
    """
    Parse one line from candidate-generation output into an arm index.
    Tries strict <Answer>...</Answer> parse first, then wraps bare text as an answer line.
    """
    if not line or not str(line).strip():
        return None
    s = str(line).strip()
    # Strip common list prefixes: "1.", "-", etc.
    s = s.lstrip("0123456789.-) ").strip()
    idx = parse_bandit_color_strict(s)
    if idx is not None:
        return idx
    wrapped = f"<answer>{s}</answer>"
    return parse_bandit_color_strict(wrapped)


@torch.no_grad()
def query_llm(system_prompt, user_prompt, temperature, max_new_tokens=50):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t = float(temperature)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if t != 0.0:
        gen_kwargs.update(do_sample=True, temperature=t, top_p=0.95)
    else:
        gen_kwargs["do_sample"] = False

    outputs = model.generate(**gen_kwargs)
    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


class LLMBanditAgent:
    """
    LLM-based bandit agent.
    Model is shared (loaded once at module import).
    Invalid parse -> (None, raw_response).
    """

    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def act(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature,
        max_new_tokens: int = 50,
    ) -> Tuple[Optional[int], str]:
        response = query_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return parse_bandit_color_strict(response), response
