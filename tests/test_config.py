"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from trafficrl.config import (
    Config,
    NetworkConfig,
    SimulationConfig,
    AgentConfig,
    RewardConfig,
    EvalConfig,
    CurriculumConfig,
    load_config,
    save_config,
)


class TestConfigDataclasses:
    """Test config dataclass defaults."""

    def test_network_config_defaults(self) -> None:
        config = NetworkConfig()
        assert config.type == "single"
        assert config.grid_size == 2
        assert config.lane_length == 100.0
        assert config.num_lanes == 1
        assert config.speed_limit == 13.89

    def test_simulation_config_defaults(self) -> None:
        config = SimulationConfig()
        assert config.steps == 3600
        assert config.step_length == 1.0
        assert config.begin == 0
        assert config.seed == 42
        assert config.gui is False

    def test_agent_config_defaults(self) -> None:
        config = AgentConfig()
        assert config.policy == "MlpPolicy"
        assert config.learning_rate == 3e-4
        assert config.total_timesteps == 100000

    def test_reward_config_defaults(self) -> None:
        config = RewardConfig()
        assert config.wait_time_weight == -1.0
        assert config.phase_change_penalty == -0.1
        assert config.queue_length_weight == 0.0

    def test_eval_config_defaults(self) -> None:
        config = EvalConfig()
        assert config.episodes == 10
        assert config.deterministic is True
        assert config.render is False

    def test_curriculum_config_defaults(self) -> None:
        config = CurriculumConfig()
        assert config.enabled is True
        assert config.stages == ["single", "double", "grid"]
        assert config.stage_timesteps == [50000, 50000, 100000]


class TestConfigLoadSave:
    """Test config loading and saving."""

    def test_save_and_load_config(self) -> None:
        """Test round-trip save/load."""
        original = Config(
            network=NetworkConfig(type="grid", grid_size=3),
            simulation=SimulationConfig(steps=1000, seed=123),
            agent=AgentConfig(total_timesteps=50000),
            reward=RewardConfig(wait_time_weight=-2.0, phase_change_penalty=-0.5),
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            save_config(original, temp_path)
            loaded = load_config(temp_path)

            assert loaded.network.type == "grid"
            assert loaded.network.grid_size == 3
            assert loaded.simulation.steps == 1000
            assert loaded.simulation.seed == 123
            assert loaded.agent.total_timesteps == 50000
            assert loaded.reward.wait_time_weight == -2.0
            assert loaded.reward.phase_change_penalty == -0.5
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_existing_config_file(self) -> None:
        """Test loading existing config files."""
        config = load_config("configs/single.yaml")
        assert config.network.type == "single"
        assert config.simulation.steps == 3600

        config = load_config("configs/double.yaml")
        assert config.network.type == "double"

        config = load_config("configs/grid.yaml")
        assert config.network.type == "grid"
        assert config.curriculum.enabled is True

    def test_config_file_not_found(self) -> None:
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_yaml_structure(self) -> None:
        """Test generated YAML has correct structure."""
        config = Config()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)
            with open(temp_path, "r") as f:
                data = yaml.safe_load(f)

            assert "network" in data
            assert "simulation" in data
            assert "agent" in data
            assert "reward" in data
            assert "eval" in data
            assert "curriculum" in data
            assert "output_dir" in data
            assert "model_save_path" in data
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])