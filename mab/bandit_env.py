import numpy as np


class BernoulliBandit:
    """K-armed Bernoulli bandit with one arm elevated by delta."""

    def __init__(self, K=5, delta=0.2, seed=None):
        self.K = K
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.best_arm = self.rng.integers(K)
        self.means = np.full(K, 0.5 - delta / 2)
        self.means[self.best_arm] = 0.5 + delta / 2
        self.optimal_mean = self.means[self.best_arm]

    def pull(self, arm: int) -> int:
        return int(self.rng.random() < self.means[arm])

    def regret(self, arm: int) -> float:
        """Instantaneous expected regret."""
        return self.optimal_mean - self.means[arm]
