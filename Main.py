# main.py
from __future__ import annotations
import time
import argparse
import csv
from typing import List

import numpy as np

from environment import GridWorldEnv
from agent import QAgent

def train(env: GridWorldEnv, agent: QAgent, episodes: int = 800) -> List[float]:
    rewards = []
    for ep in range(episodes):
        s = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            a = agent.act(s)
            s_next, r, done, _ = env.step(a)
            agent.learn(s, a, r, s_next, done)
            s = s_next
            ep_reward += r
        agent.decay_epsilon()
        rewards.append(ep_reward)
        if (ep + 1) % 50 == 0:
            print(f"[Episode {ep+1:4d}] reward={ep_reward:.1f} epsilon={agent.epsilon:.3f}")
    return rewards

def run_greedy_episode(env: GridWorldEnv, agent: QAgent, delay: float = 0.15) -> float:
    s = env.reset()
    total = 0.0
    done = False
    path_frames = []
    while not done:
        a = agent.act_greedy(s)
        s, r, done, info = env.step(a)
        total += r
        path_frames.append(env.render() + f"\nAction: {env.ACTION_NAMES[a]}\n")
    # نمایش مسیر
    print("\n--- GREEDY RUN ---")
    for f in path_frames:
        print(f)
        time.sleep(delay)
    print(f"Total reward: {total:.1f}")
    return total

def save_training_log(rewards: List[float], path: str = "training_log.csv") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards, start=1):
            w.writerow([i, float(r)])
    print(f"Saved training log to {path}")

def maybe_plot(rewards: List[float], path: str = "reward_curve.png") -> None:
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Q-Learning on GridWorld")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print(f"Saved reward plot to {path}")
    except Exception as e:
        print("Plot skipped (matplotlib not available).", e)

def main():
    parser = argparse.ArgumentParser(description="Mini Self-Driving (Grid) with Q-Learning")
    parser.add_argument("--episodes", type=int, default=800, help="training episodes")
    parser.add_argument("--delay", type=float, default=0.12, help="sleep for greedy demo (seconds)")
    args = parser.parse_args()

    env = GridWorldEnv()
    agent = QAgent(n_states=env.n_states, n_actions=env.n_actions)

    print("Training...")
    rewards = train(env, agent, episodes=args.episodes)

    agent.save("q_table.npy")
    save_training_log(rewards)
    maybe_plot(rewards)

    # تستِ greedy
    run_greedy_episode(env, agent, delay=args.delay)

if __name__ == "__main__":
    main()