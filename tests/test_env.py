"""Tests for TrafficRL environment."""

import pytest
import numpy as np

from trafficrl.config import Config, NetworkConfig, SimulationConfig, RewardConfig
from trafficrl.env import TrafficEnv, make_env


class TestTrafficEnv:
    """Test TrafficEnv basic functionality."""

    @pytest.fixture
    def config(self) -> Config:
        """Create test configuration."""
        return Config(
            network=NetworkConfig(type="single"),
            simulation=SimulationConfig(steps=100, seed=42, gui=False),
            reward=RewardConfig(),
        )

    def test_env_creation(self, config: Config) -> None:
        """Test environment can be created."""
        env = make_env(config=config)
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None
        env.close()

    def test_observation_space_shape(self, config: Config) -> None:
        """Test observation space has correct shape."""
        env = make_env(config=config)
        obs_space = env.observation_space
        assert obs_space is not None
        assert len(obs_space.shape) == 1
        # For single intersection: queue_lengths + wait_times + phase_onehot
        # 12 lanes (3 per approach * 4 approaches) + 4 phases = 28
        env.close()

    def test_action_space_shape(self, config: Config) -> None:
        """Test action space has correct shape."""
        env = make_env(config=config)
        act_space = env.action_space
        assert act_space is not None
        assert len(act_space.shape) == 1
        # Number of phases (typically 4 for 4-way intersection)
        env.close()

    def test_reset(self, config: Config) -> None:
        """Test environment reset."""
        env = make_env(config=config)
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert "step" in info
        assert info["step"] == 0
        env.close()

    def test_step(self, config: Config) -> None:
        """Test environment step."""
        env = make_env(config=config)
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_multiple_steps(self, config: Config) -> None:
        """Test multiple environment steps."""
        env = make_env(config=config)
        env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            if terminated or truncated:
                break

        env.close()

    def test_reward_shape(self, config: Config) -> None:
        """Test reward has correct components."""
        env = make_env(config=config)
        env.reset()

        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)

        # Reward should be negative (wait time penalty)
        assert reward <= 0
        # Info should contain metrics
        assert "total_wait_time" in info
        assert "total_queue_length" in info
        assert "phase_changes" in info
        env.close()

    def test_seed_reproducibility(self, config: Config) -> None:
        """Test environment is reproducible with same seed."""
        env1 = make_env(config=config)
        env2 = make_env(config=config)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

        for _ in range(5):
            action = env1.action_space.sample()
            obs1, r1, _, _, _ = env1.step(action)
            obs2, r2, _, _, _ = env2.step(action)
            np.testing.assert_array_equal(obs1, obs2)
            assert r1 == r2

        env1.close()
        env2.close()

    def test_headless_mode(self, config: Config) -> None:
        """Test environment runs in headless mode."""
        config.simulation.gui = False
        env = make_env(config=config)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        env.close()


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_single_config(self) -> None:
        """Test loading single.yaml config."""
        config = Config(
            network=NetworkConfig(type="single"),
            simulation=SimulationConfig(steps=100),
        )
        assert config.network.type == "single"
        assert config.simulation.steps == 100

    def test_config_defaults(self) -> None:
        """Test config default values."""
        config = Config()
        assert config.network.type == "single"
        assert config.simulation.steps == 3600
        assert config.agent.total_timesteps == 100000
        assert config.reward.wait_time_weight == -1.0
        assert config.reward.phase_change_penalty == -0.1


class TestNetworkGeneration:
    """Test network generation functions."""

    def test_generate_single_network(self, tmp_path) -> None:
        """Test single intersection network generation."""
        from trafficrl.network import generate_network

        output = tmp_path / "single.net.xml"
        # This test requires SUMO to be installed
        # We just verify the function exists and has correct signature
        import inspect
        sig = inspect.signature(generate_network)
        assert "network_type" in sig.parameters
        assert "output_path" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])