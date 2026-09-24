# TrafficRL

**SUMO traffic simulation + PPO agent for adaptive signal control**

TrafficRL provides a Gymnasium environment for training reinforcement learning agents to control traffic signals in SUMO (Simulation of Urban MObility). The project implements a PPO-based agent using Stable Baselines3 with curriculum learning across increasingly complex intersection configurations.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   SUMO Engine   │────▶│  Gymnasium Env   │────▶│   PPO Agent     │
│  (libsumo/TraCI)│     │  (TrafficEnv)    │     │  (SB3 PPO)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────┐         ┌──────────────┐
                       │  Observation │         │   TensorBoard│
                       │  Space:      │         │   Logging    │
                       │  [queue,     │         └──────────────┘
                       │   wait,      │
                       │   phase]     │
                       └──────────────┘
```

### Components

- **Network Generation** (`trafficrl/network.py`): Generates SUMO networks using `netgenerate` for single, double, and grid intersections
- **Environment** (`trafficrl/env.py`): Gymnasium-compliant environment with libsumo/TraCI interface
- **Configuration** (`trafficrl/config.py`): YAML-based configuration with dataclasses
- **Training** (`trafficrl/train.py`): PPO training with curriculum learning and callbacks
- **Evaluation** (`trafficrl/evaluate.py`): Model evaluation and comparison utilities
- **CLI** (`trafficrl/cli.py`): Command-line interface for all operations

## Reward Design

The reward function encourages minimizing vehicle waiting time while penalizing excessive phase changes:

```
reward = -Σ(wait_time_per_lane) - 0.1 × phase_changes
```

- **Wait time penalty**: Negative sum of accumulated waiting times across all incoming lanes
- **Phase change penalty**: Small penalty (-0.1) per phase transition to encourage smooth control
- **Configurable**: Weights adjustable via `reward.wait_time_weight` and `reward.phase_change_penalty`

## Curriculum Learning

Progressive training across intersection complexities:

| Stage | Network | Intersections | Timesteps |
|-------|---------|---------------|-----------|
| 1     | Single  | 1             | 50,000    |
| 2     | Double  | 2             | 50,000    |
| 3     | Grid    | 4             | 100,000   |

Each stage builds on the previous model, enabling transfer learning.

## TensorBoard Logging

Training metrics logged to TensorBoard:

- `rollout/ep_len_mean` - Episode length
- `rollout/ep_rew_mean` - Episode reward
- `train/loss` - Policy/value loss
- `env/total_wait_time` - Total waiting time per step
- `env/total_queue_length` - Total queue length per step
- `env/phase_changes` - Phase change count
- `curriculum/stage` - Current curriculum stage

Launch TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

## Hyperparameters (Default)

```yaml
agent:
  policy: "MlpPolicy"
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  total_timesteps: 100000
```

## Installation

```bash
# Install SUMO (Ubuntu/Debian)
sudo apt-get install sumo sumo-tools

# Install TrafficRL
pip install -e ".[dev]"
```

### Requirements

- Python 3.11+
- SUMO 1.19+ (with libsumo/TraCI)
- gymnasium ≥ 0.29
- stable-baselines3 ≥ 2.0
- pyyaml ≥ 6.0
- tensorboard ≥ 2.14
- numpy ≥ 1.24

## Quick Start

### 1. Generate Configuration
```bash
python -m trafficrl generate-config --network-type single --output configs/my_single.yaml
```

### 2. Generate Network & Routes
```bash
python -m trafficrl generate-network --network-type single --output networks/single.net.xml
python -m trafficrl create-routes --network-type single --net-file networks/single.net.xml --output routes/single.rou.xml
```

### 3. Train Agent
```bash
python -m trafficrl train --config configs/single.yaml
```

### 4. Evaluate
```bash
python -m trafficrl evaluate --config configs/single.yaml --model models/trafficrl_single.zip --episodes 10
```

### 5. View TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

## Project Structure

```
TrafficRL/
├── trafficrl/              # Main package
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── env.py              # Gymnasium environment
│   ├── evaluate.py         # Evaluation utilities
│   ├── network.py          # SUMO network generation
│   └── train.py            # PPO training with curriculum
├── configs/                # Configuration files
│   ├── single.yaml         # Single intersection
│   ├── double.yaml         # Double intersection
│   └── grid.yaml           # 2x2 grid (curriculum)
├── networks/               # Generated SUMO networks (.net.xml)
├── routes/                 # Generated route files (.rou.xml)
├── logs/                   # TensorBoard logs
├── models/                 # Saved model checkpoints
├── output/                 # Evaluation outputs
├── tests/                  # Unit tests
│   ├── test_env.py
│   └── test_config.py
├── .github/workflows/      # CI/CD
│   └── ci.yml
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── LICENSE                 # MIT License
└── .gitignore
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push/PR to main
- Installs SUMO via apt
- Runs ruff linting and format checking
- Executes pytest suite
- Performs environment smoke test
- Builds package artifact

## Results Template

After training, record results in this format:

| Stage | Timesteps | Avg Reward | Avg Wait Time (s) | Avg Queue Length | Phase Changes/ep |
|-------|-----------|------------|-------------------|------------------|------------------|
| Single| 100,000   | -XXX.XX    | XXX.X             | XX.X             | X.X              |
| Double| 100,000   | -XXX.XX    | XXX.X             | XX.X             | X.X              |
| Grid  | 200,000   | -XXX.XX    | XXX.X             | XX.X             | X.X              |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use TrafficRL in your research, please cite:

```bibtex
@software{trafficrl2024,
  title = {TrafficRL: SUMO Traffic Simulation + PPO for Adaptive Signal Control},
  author = {Kamalesh},
  year = {2024},
  url = {https://github.com/kamalesh404/TrafficRL}
}
```