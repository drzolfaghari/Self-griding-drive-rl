# agent.py
from __future__ import annotations
import numpy as np

class QAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def act_greedy(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    def learn(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        best_next = 0.0 if done else np.max(self.Q[s_next])
        target = r + self.gamma * best_next
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, path: str = "q_table.npy") -> None:
        np.save(path, self.Q)

    def load(self, path: str = "q_table.npy") -> None:
        self.Q = np.load(path)