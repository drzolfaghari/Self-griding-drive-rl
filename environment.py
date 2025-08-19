# environment.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

Action = int  # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT

class GridWorldEnv:
    """
    یک محیط ساده Grid برای تمرین Q-Learning.
    A: عامل (ماشین)، S: شروع، G: هدف، #: مانع
    """

    ACTIONS: Dict[Action, Tuple[int, int]] = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        start: Tuple[int, int] = (5, 0),
        goal: Tuple[int, int] = (0, 5),
        obstacles: Tuple[Tuple[int, int], ...] = ((1, 1), (1, 2), (2, 2), (3, 1), (4, 3)),
        step_cost: float = -1.0,
        wall_cost: float = -5.0,
        goal_reward: float = 25.0,
        slip_prob: float = 0.10,
        max_steps: int = 200,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)
        self.step_cost = step_cost
        self.wall_cost = wall_cost
        self.goal_reward = goal_reward
        self.slip_prob = slip_prob
        self.max_steps = max_steps

        self.n_states = width * height
        self.n_actions = 4

        self.rng = np.random.default_rng(seed)
        self._pos = tuple(start)
        self._steps = 0

    # --- helpers ---
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def _valid_cell(self, r: int, c: int) -> bool:
        return self._in_bounds(r, c) and (r, c) not in self.obstacles

    def _encode(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * self.width + c

    def _decode(self, s: int) -> Tuple[int, int]:
        return (s // self.width, s % self.width)

    # --- public API ---
    def reset(self) -> int:
        self._pos = tuple(self.start)
        self._steps = 0
        return self._encode(self._pos)

    def step(self, action: Action):
        self._steps += 1

        # slip: با احتمال slip_prob عمل تصادفی رخ می‌دهد
        if self.rng.random() < self.slip_prob:
            action = int(self.rng.integers(0, self.n_actions))

        dr, dc = self.ACTIONS[action]
        nr, nc = self._pos[0] + dr, self._pos[1] + dc

        reward = self.step_cost
        done = False

        if not self._valid_cell(nr, nc):
            # برخورد با دیوار یا مانع: سر جای قبلی می‌مانیم و جریمه می‌گیریم
            reward += self.wall_cost
            next_pos = self._pos
        else:
            next_pos = (nr, nc)

        if next_pos == self.goal:
            reward += self.goal_reward
            done = True

        if self._steps >= self.max_steps:
            done = True

        self._pos = next_pos
        s_next = self._encode(self._pos)
        info = {"pos": self._pos, "steps": self._steps}
        return s_next, float(reward), bool(done), info

    def render(self) -> str:
        """یک رشته‌ی ASCII از محیط برمی‌گرداند (برای پرینت)."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for (r, c) in self.obstacles:
            grid[r][c] = "#"
        sr, sc = self.start
        gr, gc = self.goal
        grid[sr][sc] = "S"
        grid[gr][gc] = "G"
        ar, ac = self._pos
        grid[ar][ac] = "A"

        lines = [" ".join(row) for row in grid]
        return "\n".join(lines)

    @property
    def state(self) -> int:
        return self._encode(self._pos)