"""Token accounting for ``LambdaAutonomousAgent._score_actions`` (HF scorer).

The scorer tokenizes ``prompt + action`` as a single string per candidate, with
micro-batching and padding — so token counts must mirror that encoding, not
``len(encode(prompt)) + len(encode(action))`` (boundary effects differ).

Use :func:`compute_scoring_forward_token_stats` for the same numbers the model
actually processes (sum of non-padding positions per forward).
"""

from typing import Any, Dict, List, Sequence


def compute_scoring_forward_token_stats(
    tokenizer,
    prompt: str,
    actions: Sequence[str],
    micro_batch_size: int,
) -> Dict[str, Any]:
    """
    Mirror ``LambdaAutonomousAgent._score_actions`` encoding without running the model.

    Returns token lengths that match each batched ``tokenizer(..., padding=True)``
    forward: for each sequence, non-padding length is ``attention_mask.sum()``.

    Keys
    ----
    prompt_token_len
        Length of ``tokenizer(prompt, ...)`` (same convention as the agent).
    num_candidates
        Number of actions scored.
    per_sequence_tokens
        List of non-padding token counts for each ``prompt + action`` string.
    total_forward_tokens
        Sum of ``per_sequence_tokens`` — total real tokens processed across all
        micro-batch forwards (each token position is computed once per forward).
    micro_batches
        One entry per micro-batch: ``batch_size``, ``tokens_non_pad`` (sum for that batch).
    """
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=8192
    )["input_ids"]
    prompt_len = int(prompt_ids.shape[1])

    per_sequence_tokens = []  # type: List[int]
    micro_batches = []  # type: List[Dict[str, Any]]

    for i in range(0, len(actions), micro_batch_size):
        batch_actions = actions[i : i + micro_batch_size]
        full_texts = [prompt + action for action in batch_actions]
        enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )
        attention_mask = enc["attention_mask"]
        batch_non_pad = int(attention_mask.sum().item())
        for b in range(attention_mask.size(0)):
            full_len = int(attention_mask[b].sum().item())
            per_sequence_tokens.append(full_len)
        micro_batches.append(
            {
                "start_index": i,
                "batch_size": len(batch_actions),
                "tokens_non_pad": batch_non_pad,
            }
        )

    total_forward = sum(per_sequence_tokens)

    return {
        "prompt_token_len": prompt_len,
        "num_candidates": len(actions),
        "per_sequence_tokens": per_sequence_tokens,
        "total_forward_tokens": total_forward,
        "micro_batches": micro_batches,
        "micro_batch_size_config": micro_batch_size,
        "note": (
            "total_forward_tokens sums non-pad tokens per prompt+action sequence, "
            "matching the scorer's tokenizer calls; use for cost/load estimates."
        ),
    }
