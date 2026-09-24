"""Configuration loading for TrafficRL."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class NetworkConfig:
    """Network configuration."""
    type: str = "single"  # single, double, grid
    grid_size: int = 2
    lane_length: float = 100.0
    num_lanes: int = 1
    speed_limit: float = 13.89
    network_file: Optional[str] = None


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    steps: int = 3600
    step_length: float = 1.0
    begin: int = 0
    seed: int = 42
    gui: bool = False


@dataclass
class AgentConfig:
    """PPO agent configuration."""
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 100000
    tensorboard_log: str = "logs/tensorboard"


@dataclass
class RewardConfig:
    """Reward function configuration."""
    wait_time_weight: float = -1.0
    phase_change_penalty: float = -0.1
    queue_length_weight: float = 0.0


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    episodes: int = 10
    deterministic: bool = True
    render: bool = False


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = True
    stages: List[str] = field(default_factory=lambda: ["single", "double", "grid"])
    stage_timesteps: List[int] = field(default_factory=lambda: [50000, 50000, 100000])
    stage_thresholds: Optional[List[float]] = None  # Optional performance thresholds


@dataclass
class Config:
    """Main configuration class."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    output_dir: str = "output"
    model_save_path: str = "models/trafficrl_model.zip"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with loaded values
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    return _dict_to_config(data)


def _dict_to_config(data: Dict[str, Any]) -> Config:
    """Convert dictionary to Config object."""
    config = Config()

    if "network" in data:
        config.network = NetworkConfig(**data["network"])
    if "simulation" in data:
        config.simulation = SimulationConfig(**data["simulation"])
    if "agent" in data:
        config.agent = AgentConfig(**data["agent"])
    if "reward" in data:
        config.reward = RewardConfig(**data["reward"])
    if "eval" in data:
        config.eval = EvalConfig(**data["eval"])
    if "curriculum" in data:
        config.curriculum = CurriculumConfig(**data["curriculum"])
    if "output_dir" in data:
        config.output_dir = data["output_dir"]
    if "model_save_path" in data:
        config.model_save_path = data["model_save_path"]

    return config


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "network": config.network.__dict__,
        "simulation": config.simulation.__dict__,
        "agent": config.agent.__dict__,
        "reward": config.reward.__dict__,
        "eval": config.eval.__dict__,
        "curriculum": config.curriculum.__dict__,
        "output_dir": config.output_dir,
        "model_save_path": config.model_save_path,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)