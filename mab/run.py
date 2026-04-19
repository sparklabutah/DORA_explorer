

from __future__ import annotations

import argparse
import json
import os
import random
import socket
from datetime import datetime

import numpy as np

from bandit_env import BernoulliBandit
from agents.baselines import EpsilonGreedyAgent, GreedyAgent, ThompsonSamplingAgent, UCBAgent
from evaluation import (
    best_arm_fraction,
    greedy_fraction,
    greedy_fraction_sparse,
    min_frac,
    suffix_failure,
    suffix_failure_sparse,
)
from prompts import ARM_NAMES, raw_history_prompt, summarized_history_prompt, system_prompt

# ---------------------------------------------------------------------------
# Logging helpers (inlined — no external log_utils dependency)
# ---------------------------------------------------------------------------


def _create_log_dir(base_dir: str = "logs", prefix: str = "BMAB_run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_id = os.getenv("SLURM_JOB_ID")
    tag = f"{prefix}_{slurm_id or socket.gethostname()}_{timestamp}_{random.randint(1000, 9999)}"
    logdir = os.path.join(base_dir, tag)
    os.makedirs(logdir, exist_ok=True)
    return logdir


def _write_jsonl(logdir: str, filename: str, entry: dict) -> None:
    with open(os.path.join(logdir, filename), "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Shared parser helpers
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=None, help="Number of episodes (default depends on agent)")
    parser.add_argument("--K", type=int, default=5, help="Number of arms")
    parser.add_argument("--delta", type=float, default=0.2, help="Best-arm gap")


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def _run_baselines(args: argparse.Namespace) -> None:
    n_rep = args.replicates or 1000
    K, delta, HORIZON = args.K, args.delta, args.horizon

    agents = {
        "UCB": lambda: UCBAgent(K),
        "TS": lambda: ThompsonSamplingAgent(K),
        "Greedy": lambda: GreedyAgent(K),
        "EpsilonGreedy": lambda: EpsilonGreedyAgent(K, 0.1),
    }

    logdir = _create_log_dir(prefix="BMAB_baselines")
    print(f"[INFO] Logging to {logdir}")

    meta = {"K": K, "delta": delta, "HORIZON": HORIZON, "N_REPLICATES": n_rep, "agents": list(agents.keys())}
    with open(os.path.join(logdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    out_dir = os.path.join(logdir, "baselines")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Running baselines {list(agents.keys())} ({n_rep} episodes x {HORIZON} steps)")

    for episode in range(n_rep):
        for agent_name, agent_fn in agents.items():
            env = BernoulliBandit(K=K, delta=delta, seed=episode)
            agent = agent_fn()
            history, actions, regrets = [], [], []

            for step in range(HORIZON):
                action = agent.act()
                reward = env.pull(action)
                regret = env.regret(action)
                agent.update(action, reward)

                history.append((action, reward))
                actions.append(action)
                regrets.append(regret)

                _write_jsonl(out_dir, f"{agent_name}.jsonl", {
                    "episode": episode, "step": step, "agent": agent_name,
                    "action": int(action), "action_name": ARM_NAMES[action],
                    "reward": int(reward), "instant_regret": float(regret),
                    "cumulative_regret": float(sum(regrets)),
                    "best_arm": int(env.best_arm),
                    "is_best_arm": bool(action == env.best_arm),
                })

            _write_jsonl(out_dir, f"{agent_name}.jsonl", {"episode_summary": {
                "episode": episode, "agent": agent_name,
                "suffix_failure": bool(suffix_failure(actions, env.best_arm, HORIZON // 2)),
                "min_frac": float(min_frac(np.array(actions, dtype=int), K)),
                "best_arm_fraction": float(best_arm_fraction(actions, env.best_arm)),
                "greedy_fraction": float(greedy_fraction(
                    np.array(actions, dtype=int),
                    np.array([r for _, r in history], dtype=float), K)),
                "total_reward": int(sum(r for _, r in history)),
                "total_regret": float(sum(regrets)),
            }})

    print("[INFO] Baselines finished.")


# ---------------------------------------------------------------------------
# dora (lambda policy)
# ---------------------------------------------------------------------------


def _run_dora(args: argparse.Namespace) -> None:
    from agents.dora_lambda_schedule import LambdaPolicyLLMAgent

    n_rep = args.replicates or 20
    K, delta, HORIZON = args.K, args.delta, args.horizon
    USE_SUMMARY = not args.raw_history

    sys_prompt = system_prompt(HORIZON)

    logdir = _create_log_dir(prefix="LambdaPolicy_run")
    print(f"[INFO] Logging to {logdir} | alpha={args.alpha}, beta={args.beta}")

    meta = {
        "K": K, "delta": delta, "HORIZON": HORIZON, "N_REPLICATES": n_rep,
        "USE_SUMMARY": USE_SUMMARY,
        "ALPHA": args.alpha, "BETA": args.beta,
        "LAMBDA_START": args.lambda_start, "LAMBDA_END": args.lambda_end,
        "LAMBDA_K": args.lambda_k,
        "NUM_CANDIDATES": args.num_candidates, "GEN_TEMP": args.gen_temp,
        "HF_MODEL": os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    }
    with open(os.path.join(logdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for episode in range(n_rep):
        env = BernoulliBandit(K=K, delta=delta, seed=episode)
        agent = LambdaPolicyLLMAgent(
            horizon=HORIZON, alpha=args.alpha, beta=args.beta,
            lambda_start=args.lambda_start, lambda_end=args.lambda_end,
            lambda_k=args.lambda_k, micro_batch_size=2,
            num_candidates=args.num_candidates, gen_temp=args.gen_temp,
            seed=episode,
        )

        history, actions, regrets = [], [], []

        for step in range(HORIZON):
            user_prompt = (
                summarized_history_prompt(history) if USE_SUMMARY else raw_history_prompt(history)
            )
            action_idx, diagnostics, probs = agent.act(sys_prompt, user_prompt)

            reward = env.pull(action_idx)
            regret = env.regret(action_idx)
            history.append((action_idx, reward))
            actions.append(action_idx)
            regrets.append(regret)

            _write_jsonl(logdir, "LambdaPolicy.jsonl", {
                "episode": episode, "step": step, "agent": "LambdaPolicy",
                "lambda_value": agent.current_lambda(),
                "action": int(action_idx), "action_name": ARM_NAMES[action_idx],
                "reward": int(reward), "instant_regret": float(regret),
                "cumulative_regret": float(sum(regrets)),
                "best_arm": int(env.best_arm),
                "is_best_arm": bool(action_idx == env.best_arm),
            })

        _write_jsonl(logdir, "LambdaPolicy.jsonl", {"episode_summary": {
            "episode": episode, "agent": "LambdaPolicy",
            "suffix_failure": bool(suffix_failure(actions, env.best_arm, HORIZON // 2)),
            "min_frac": float(min_frac(actions, K)),
            "best_arm_fraction": float(best_arm_fraction(actions, env.best_arm)),
            "greedy_fraction": float(greedy_fraction(
                np.array(actions, dtype=int),
                np.array([r for _, r in history], dtype=float), K)),
            "total_reward": int(sum(r for _, r in history)),
            "total_regret": float(sum(regrets)),
        }})

    print("[INFO] Lambda Policy experiment finished.")


# ---------------------------------------------------------------------------
# temperature-sweep
# ---------------------------------------------------------------------------


def _run_temp_sweep(args: argparse.Namespace) -> None:
    from agents.llm import HF_MODEL, LLMBanditAgent

    n_rep = args.replicates or 20
    K, delta, HORIZON = args.K, args.delta, args.horizon
    USE_SUMMARY = not args.raw_history
    TEMPERATURES = args.temperatures
    MAX_NEW_TOKENS = args.max_new_tokens
    INVALID_REGRET = args.invalid_regret

    logdir = _create_log_dir(prefix="BMAB_temp_sweep")
    print(f"[INFO] Root log directory: {logdir}")

    meta = {
        "K": K, "delta": delta, "HORIZON": HORIZON, "N_REPLICATES": n_rep,
        "USE_SUMMARY": USE_SUMMARY, "HF_MODEL": HF_MODEL,
        "temperatures": TEMPERATURES, "max_new_tokens": MAX_NEW_TOKENS,
        "invalid_step_regret": INVALID_REGRET,
    }
    with open(os.path.join(logdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    sys_p = system_prompt(HORIZON)
    agent = LLMBanditAgent()
    print(f"Model: {HF_MODEL}")
    print(f"{len(TEMPERATURES)} temperatures x {n_rep} replicates x {HORIZON} steps\n")

    for temperature in TEMPERATURES:
        temp_dir = os.path.join(logdir, f"T_{temperature}")
        os.makedirs(temp_dir, exist_ok=True)
        jsonl_name = "LLMBanditAgent.jsonl"

        folder_meta = {**meta, "temperature": float(temperature), "folder": f"T_{temperature}"}
        with open(os.path.join(temp_dir, "meta.json"), "w") as f:
            json.dump(folder_meta, f, indent=2)

        print(f"[INFO] T={temperature}: {n_rep} episodes x {HORIZON} steps -> {temp_dir}")

        for episode in range(n_rep):
            env = BernoulliBandit(K=K, delta=delta, seed=episode)
            history, regrets, actions_valid = [], [], []
            actions_series = [None] * HORIZON
            rewards_series = [None] * HORIZON

            for step in range(HORIZON):
                user_prompt = (
                    summarized_history_prompt(history) if USE_SUMMARY else raw_history_prompt(history)
                )
                action, raw_response = agent.act(
                    sys_p, user_prompt, temperature=temperature, max_new_tokens=MAX_NEW_TOKENS
                )

                if action is None:
                    regret = INVALID_REGRET
                    regrets.append(regret)
                    actions_series[step] = None
                    rewards_series[step] = 0
                    log_entry = {
                        "episode": episode, "step": step, "agent": "LLMBanditAgent",
                        "valid": False, "temperature": float(temperature),
                        "action": None, "action_name": None,
                        "reward": 0, "instant_regret": float(regret),
                        "cumulative_regret": float(sum(regrets)),
                        "best_arm": int(env.best_arm), "is_best_arm": None,
                        "raw_response": raw_response,
                    }
                else:
                    reward = env.pull(action)
                    regret = env.regret(action)
                    history.append((action, reward))
                    actions_series[step] = int(action)
                    rewards_series[step] = int(reward)
                    regrets.append(regret)
                    actions_valid.append(int(action))
                    log_entry = {
                        "episode": episode, "step": step, "agent": "LLMBanditAgent",
                        "valid": True, "temperature": float(temperature),
                        "action": int(action), "action_name": ARM_NAMES[action],
                        "reward": int(reward), "instant_regret": float(regret),
                        "cumulative_regret": float(sum(regrets)),
                        "best_arm": int(env.best_arm),
                        "is_best_arm": bool(action == env.best_arm),
                        "raw_response": raw_response,
                    }

                _write_jsonl(temp_dir, jsonl_name, log_entry)

            n_valid = len(actions_valid)
            _write_jsonl(temp_dir, jsonl_name, {"episode_summary": {
                "episode": episode, "agent": "LLMBanditAgent",
                "temperature": float(temperature),
                "valid_pulls": n_valid,
                "invalid_steps": HORIZON - n_valid,
                "suffix_failure": bool(suffix_failure_sparse(actions_series, env.best_arm, HORIZON // 2)),
                "min_frac": float(min_frac(np.array(actions_valid, dtype=int), K)) if n_valid > 0 else float("nan"),
                "best_arm_fraction": float(best_arm_fraction(actions_valid, env.best_arm)) if n_valid > 0 else float("nan"),
                "greedy_fraction": float(greedy_fraction_sparse(actions_series, rewards_series, K)),
                "total_reward": int(sum(r for _, r in history)),
                "total_regret": float(sum(regrets)),
            }})

    print(f"[INFO] Finished. All logs under {logdir}")


# ---------------------------------------------------------------------------
# scheduled-temp (exponential temperature schedule)
# ---------------------------------------------------------------------------


def _run_scheduled_temp(args: argparse.Namespace) -> None:
    from agents.llm import HF_MODEL
    from agents.scheduled_temp import ScheduledTempLLMAgent

    n_rep = args.replicates or 20
    K, delta, HORIZON = args.K, args.delta, args.horizon
    USE_SUMMARY = not args.raw_history
    INVALID_REGRET = args.invalid_regret

    logdir = _create_log_dir(prefix="BMAB_scheduled_temp")
    print(f"[INFO] Logging to {logdir}")

    meta = {
        "K": K, "delta": delta, "HORIZON": HORIZON, "N_REPLICATES": n_rep,
        "USE_SUMMARY": USE_SUMMARY, "HF_MODEL": HF_MODEL,
        "temp_start": args.temp_start, "temp_end": args.temp_end,
        "schedule_k": args.schedule_k,
        "max_new_tokens": args.max_new_tokens,
        "invalid_step_regret": INVALID_REGRET,
    }
    with open(os.path.join(logdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    sys_p = system_prompt(HORIZON)
    jsonl_name = "ScheduledTempLLM.jsonl"

    print(f"Model: {HF_MODEL}")
    print(
        f"temp {args.temp_start} → {args.temp_end} (k={args.schedule_k}) | "
        f"{n_rep} replicates x {HORIZON} steps\n"
    )

    for episode in range(n_rep):
        env = BernoulliBandit(K=K, delta=delta, seed=episode)
        agent = ScheduledTempLLMAgent(
            horizon=HORIZON,
            temp_start=args.temp_start,
            temp_end=args.temp_end,
            schedule_k=args.schedule_k,
            max_new_tokens=args.max_new_tokens,
        )

        history, regrets, actions_valid = [], [], []
        actions_series = [None] * HORIZON
        rewards_series = [None] * HORIZON

        for step in range(HORIZON):
            user_prompt = (
                summarized_history_prompt(history) if USE_SUMMARY else raw_history_prompt(history)
            )
            action, raw_response = agent.act(sys_p, user_prompt)
            cur_temp = agent.current_temperature()

            if action is None:
                regret = INVALID_REGRET
                regrets.append(regret)
                actions_series[step] = None
                rewards_series[step] = 0
                log_entry = {
                    "episode": episode, "step": step, "agent": "ScheduledTempLLM",
                    "valid": False, "temperature": cur_temp,
                    "action": None, "action_name": None,
                    "reward": 0, "instant_regret": float(regret),
                    "cumulative_regret": float(sum(regrets)),
                    "best_arm": int(env.best_arm), "is_best_arm": None,
                    "raw_response": raw_response,
                }
            else:
                reward = env.pull(action)
                regret = env.regret(action)
                history.append((action, reward))
                actions_series[step] = int(action)
                rewards_series[step] = int(reward)
                regrets.append(regret)
                actions_valid.append(int(action))
                log_entry = {
                    "episode": episode, "step": step, "agent": "ScheduledTempLLM",
                    "valid": True, "temperature": cur_temp,
                    "action": int(action), "action_name": ARM_NAMES[action],
                    "reward": int(reward), "instant_regret": float(regret),
                    "cumulative_regret": float(sum(regrets)),
                    "best_arm": int(env.best_arm),
                    "is_best_arm": bool(action == env.best_arm),
                    "raw_response": raw_response,
                }

            _write_jsonl(logdir, jsonl_name, log_entry)

        n_valid = len(actions_valid)
        _write_jsonl(logdir, jsonl_name, {"episode_summary": {
            "episode": episode, "agent": "ScheduledTempLLM",
            "valid_pulls": n_valid,
            "invalid_steps": HORIZON - n_valid,
            "suffix_failure": bool(suffix_failure_sparse(actions_series, env.best_arm, HORIZON // 2)),
            "min_frac": float(min_frac(np.array(actions_valid, dtype=int), K)) if n_valid > 0 else float("nan"),
            "best_arm_fraction": float(best_arm_fraction(actions_valid, env.best_arm)) if n_valid > 0 else float("nan"),
            "greedy_fraction": float(greedy_fraction_sparse(actions_series, rewards_series, K)),
            "total_reward": int(sum(r for _, r in history)),
            "total_regret": float(sum(regrets)),
        }})

    print(f"[INFO] Scheduled-temp finished. Logs under {logdir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified MAB experiment runner (baselines / DORA / scheduled-temp / temperature-sweep).",
    )
    sub = parser.add_subparsers(dest="agent", required=True)

    # -- baselines --
    p_bl = sub.add_parser("baselines", help="UCB, Thompson Sampling, Greedy, ε-Greedy")
    _add_common_args(p_bl)

    # -- dora --
    p_dora = sub.add_parser("dora", help="DORA lambda policy (generate → score → λ-softmax)")
    _add_common_args(p_dora)
    p_dora.add_argument("--alpha", type=float, default=0.8)
    p_dora.add_argument("--beta", type=float, default=0.2)
    p_dora.add_argument("--lambda-start", type=float, default=0.0)
    p_dora.add_argument("--lambda-end", type=float, default=40.0)
    p_dora.add_argument("--lambda-k", type=float, default=7.0)
    p_dora.add_argument("--num-candidates", type=int, default=10)
    p_dora.add_argument("--gen-temp", type=float, default=0.7)
    p_dora.add_argument("--raw-history", action="store_true", help="Use raw history prompt instead of summary")

    # -- scheduled-temp --
    p_st = sub.add_parser("scheduled-temp", help="Exponential temperature schedule (high→low) LLM agent")
    _add_common_args(p_st)
    p_st.add_argument("--temp-start", type=float, default=2.0, help="Starting temperature")
    p_st.add_argument("--temp-end", type=float, default=0.0, help="Final temperature")
    p_st.add_argument("--schedule-k", type=float, default=5.0, help="Exponential schedule steepness")
    p_st.add_argument("--max-new-tokens", type=int, default=200)
    p_st.add_argument("--invalid-regret", type=float, default=0.2, help="Regret charged for unparseable LLM output")
    p_st.add_argument("--raw-history", action="store_true", help="Use raw history prompt instead of summary")

    # -- temperature-sweep --
    p_ts = sub.add_parser("temperature-sweep", help="Fixed-temperature sweep for zero-shot LLM agent")
    _add_common_args(p_ts)
    p_ts.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.3, 0.7, 1.0, 1.5, 2.0])
    p_ts.add_argument("--max-new-tokens", type=int, default=200)
    p_ts.add_argument("--invalid-regret", type=float, default=0.2, help="Regret charged for unparseable LLM output")
    p_ts.add_argument("--raw-history", action="store_true", help="Use raw history prompt instead of summary")

    args = parser.parse_args()

    dispatch = {
        "baselines": _run_baselines,
        "dora": _run_dora,
        "scheduled-temp": _run_scheduled_temp,
        "temperature-sweep": _run_temp_sweep,
    }
    dispatch[args.agent](args)


if __name__ == "__main__":
    main()
