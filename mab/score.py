"""Sequence-level scoring: mean token log-probability and variance for each candidate response."""

import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("SCORING_MODEL", os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))


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


def score_responses_same_prompt(
    prompt, responses, max_length=100000, micro_batch_size=2, alpha=1.0, beta=0.0
):
    """
    Score candidate responses sharing the same prompt.

    For each response, computes:
    - mean_log_prob: mean per-token log probability
    - variance: variance of per-token log probabilities
    - final_score: alpha * norm_mean - beta * norm_variance (min-max normalized)
    """
    results = []

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    )["input_ids"].to(model.device)
    prompt_len = prompt_ids.shape[1]
    model.eval()

    raw_stats = []

    for i in range(0, len(responses), micro_batch_size):
        batch_responses = responses[i : i + micro_batch_size]
        full_texts = [prompt + r for r in batch_responses]

        enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)

        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        for b in range(input_ids.size(0)):
            full_len = attention_mask[b].sum().item()
            resp_start = prompt_len
            label_start = resp_start - 1
            resp_len = full_len - resp_start

            if resp_len <= 0:
                raw_stats.append(
                    {
                        "token_log_probs": [],
                        "cumulative_log_prob": 0.0,
                        "mean_log_prob": 0.0,
                        "variance": 0.0,
                    }
                )
                continue

            logit_slice = logits[b, label_start : label_start + resp_len]
            label_slice = labels[b, label_start : label_start + resp_len]

            log_probs = F.log_softmax(logit_slice, dim=-1)
            token_log_probs = log_probs.gather(1, label_slice.unsqueeze(-1)).squeeze(-1)

            cumulative = token_log_probs.sum().item()
            mean_log_prob = token_log_probs.mean().item()
            variance = token_log_probs.var(unbiased=False).item()

            raw_stats.append(
                {
                    "token_log_probs": token_log_probs.tolist(),
                    "cumulative_log_prob": cumulative,
                    "mean_log_prob": mean_log_prob,
                    "variance": variance,
                }
            )

    # Min-max normalize and compute final_score
    log_means = [r["mean_log_prob"] for r in raw_stats]
    variances = [r["variance"] for r in raw_stats]

    min_neg, max_neg = min(log_means), max(log_means)
    min_var, max_var = min(variances), max(variances)
    eps = 1e-8

    for r in raw_stats:
        if max_neg == min_neg:
            norm_log_mean = 0.5
        else:
            norm_log_mean = (r["mean_log_prob"] - min_neg) / (max_neg - min_neg + eps)

        if max_var == min_var:
            norm_var = 0.5
        else:
            norm_var = (r["variance"] - min_var) / (max_var - min_var + eps)

        r["norm_mean_log_prob"] = norm_log_mean
        r["norm_variance"] = norm_var
        r["final_score"] = alpha * norm_log_mean - beta * norm_var
        results.append(r)

    return results
