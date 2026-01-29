# Multidrone

PPO-based reinforcement learning for quadrotor drone flight control using MuJoCo physics simulation and Stable-Baselines3.

## Overview

A custom Gymnasium environment simulates a quadrotor with 4 thrust actuators in MuJoCo. A PPO agent learns to stabilize and navigate the drone to a target hover position through reward shaping on position, orientation, and crash penalties.

## Environment

| Property | Details |
|----------|---------|
| **Action space** | Continuous, 4D (per-rotor thrust commands) |
| **Observation space** | 6D (x, y, z position + roll, pitch, yaw) |
| **Target** | Hover at (0, 0, 0.5) |
| **Max steps** | 500 per episode |
| **Physics** | MuJoCo with 4 thrust + 4 torque actuators |

## Requirements

```
gymnasium
stable_baselines3
mujoco
mujoco_viewer
scipy
numpy
torch
pybullet
```

Install with:
```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train with a single environment
python ppo_train.py

# Train with 8 parallel environments (faster)
python train_drone_parallel.py

# Test a trained model with visualization
python testing.py
```

Trained models are saved to `agents/`.

## Project Structure

```
multidrone/
├── env.py                    # Custom Gymnasium quadrotor environment
├── ppo_train.py              # Single-environment PPO training
├── train_drone_parallel.py   # Parallel training (8x SubprocVecEnv)
├── testing.py                # Model evaluation with rendering
├── multi_drone_env.py        # Multi-drone environment (WIP)
├── 3D/                       # MuJoCo model files (MJCF/XML)
├── agents/                   # Saved model checkpoints
├── assets/                   # Additional resources
├── requirements.txt
└── README.md
```

## Training Details

- **Algorithm:** PPO (Proximal Policy Optimization)
- **Timesteps:** 200k default
- **Rollout:** 2048 steps per update
- **Parallelization:** Optional 8-environment SubprocVecEnv for faster training
- **Reward:** Position error penalty + orientation penalty + crash penalty + time penalty
