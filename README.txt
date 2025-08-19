# Mini Self-Driving (Grid) with Q-Learning

A tiny **GridWorld** where an agent (the "car") learns to reach the goal while avoiding obstacles using **Q-Learning**.

## Features
- Clean `environment.py` with obstacles, slip/noise, step/wall costs
- `QAgent` with epsilon-greedy, decay, save/load
- Training log (`training_log.csv`) + optional reward plot (`reward_curve.png`)
- Greedy demo with ASCII render

## How to Run
```bash
# 1) Clone
git clone https://github.com/<your-username>/self-driving-grid-rl.git
cd self-driving-grid-rl

# 2) Install
pip install -r requirements.txt

# 3) Train + plot + greedy demo
python main.py --episodes 800