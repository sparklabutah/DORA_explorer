import numpy as np


def cumulative_regret(regrets):
    return np.cumsum(regrets)


def suffix_failure(actions, best_arm, t0):
    """Best arm never pulled after time t0."""
    return best_arm not in actions[t0:]


def suffix_failure_sparse(actions_series, best_arm, t0):
    """Like suffix_failure, but actions_series may contain None for invalid steps."""
    for a in actions_series[t0:]:
        if a is not None and int(a) == int(best_arm):
            return False
    return True


def min_frac(actions, K):
    """Uniform-like failure metric: fraction of the least-pulled arm."""
    counts = np.bincount(actions, minlength=K)
    return np.min(counts) / len(actions)


def best_arm_fraction(actions, best_arm):
    return np.mean([a == best_arm for a in actions])


def greedy_fraction(actions, rewards, K):
    """
    Fraction of rounds in which the agent selected a greedy arm
    (one with the highest empirical mean so far).
    """
    counts = np.zeros(K)
    means = np.zeros(K)
    T = len(actions)
    greedy_rounds = 0

    for t in range(T):
        if np.sum(counts) > 0:
            max_mean = np.max(means)
            greedy_arms = np.where(means == max_mean)[0]
        else:
            greedy_arms = np.arange(K)

        if actions[t] in greedy_arms:
            greedy_rounds += 1

        a = actions[t]
        counts[a] += 1
        means[a] += (rewards[t] - means[a]) / counts[a]

    return greedy_rounds / T


def greedy_fraction_sparse(actions_series, rewards_series, K):
    """Like greedy_fraction; None entries are skipped (invalid LLM step)."""
    counts = np.zeros(K)
    means = np.zeros(K)
    greedy_rounds = 0
    n_valid = 0

    for t in range(len(actions_series)):
        a = actions_series[t]
        if a is None:
            continue
        r = rewards_series[t]
        n_valid += 1
        if np.sum(counts) > 0:
            max_mean = np.max(means)
            greedy_arms = np.where(means == max_mean)[0]
        else:
            greedy_arms = np.arange(K)
        ai = int(a)
        if ai in greedy_arms:
            greedy_rounds += 1
        counts[ai] += 1
        means[ai] += (float(r) - means[ai]) / counts[ai]

    if n_valid == 0:
        return float("nan")
    return greedy_rounds / n_valid
