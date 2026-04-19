import numpy as np


class UCBAgent:
    def __init__(self, K, C=1.0):
        self.K = K
        self.C = C
        self.counts = np.zeros(K)
        self.means = np.zeros(K)

    def act(self):
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
        ucb = self.means + np.sqrt(self.C / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n


class ThompsonSamplingAgent:
    """Thompson Sampling with Beta-Bernoulli prior (Uniform[0,1] default)."""

    def __init__(self, K, alpha0=1.0, beta0=1.0):
        self.K = K
        self.alpha = np.ones(K) * alpha0
        self.beta = np.ones(K) * beta0

    def act(self):
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


class GreedyAgent:
    """Pure greedy: sample each arm once, then always pick highest empirical mean."""

    def __init__(self, K):
        self.K = K
        self.counts = np.zeros(K)
        self.means = np.zeros(K)
        self.initialized = False

    def act(self):
        if not self.initialized:
            for a in range(self.K):
                if self.counts[a] == 0:
                    return a
            self.initialized = True
        return int(np.argmax(self.means))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n


class EpsilonGreedyAgent:
    def __init__(self, k, epsilon, epsilon_min=0.01, epsilon_decay=0.99):
        self.k = k
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
