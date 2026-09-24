"""Gymnasium environment for SUMO traffic signal control."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import sumolib
import traci
from gymnasium import spaces

from trafficrl.config import Config, NetworkConfig, RewardConfig, SimulationConfig


class TrafficEnv(gym.Env):
    """
    Gymnasium environment for adaptive traffic signal control using SUMO.

    Observation space:
        - queue_lengths: Number of vehicles waiting at each incoming lane
        - wait_times: Accumulated waiting time for vehicles at each incoming lane
        - current_phase: Current traffic light phase (one-hot encoded)

    Action space:
        - phase_duration: Duration for the next phase (10-60 seconds)

    Reward:
        - Negative sum of waiting times across all lanes
        - Penalty for phase changes to encourage smooth control
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[str] = None,
        network_config: Optional[NetworkConfig] = None,
        simulation_config: Optional[SimulationConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        sumo_binary: Optional[str] = None,
        net_file: Optional[str] = None,
        route_file: Optional[str] = None,
        sumocfg_file: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the traffic environment.

        Args:
            config: Full Config object (alternative to individual configs)
            config_path: Path to YAML config file
            network_config: Network configuration
            simulation_config: Simulation configuration
            reward_config: Reward configuration
            sumo_binary: SUMO binary to use (sumo or sumo-gui)
            net_file: Path to .net.xml file
            route_file: Path to .rou.xml file
            sumocfg_file: Path to .sumocfg file
            render_mode: Render mode ('human' or 'rgb_array')
        """
        super().__init__()

        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = Config(
                network=network_config or NetworkConfig(),
                simulation=simulation_config or SimulationConfig(),
                reward=reward_config or RewardConfig(),
            )

        self.render_mode = render_mode
        self.sumo_binary = sumo_binary or ("sumo-gui" if self.config.simulation.gui else "sumo")

        # Network files
        self.net_file = net_file
        self.route_file = route_file
        self.sumocfg_file = sumocfg_file

        # SUMO connection
        self.conn: Optional[traci.Connection] = None
        self.tls_ids: List[str] = []
        self.incoming_lanes: Dict[str, List[str]] = {}
        self.phase_definitions: Dict[str, List[Dict]] = {}

        # Environment state
        self.current_step = 0
        self.total_steps = self.config.simulation.steps
        self.last_phase: Dict[str, int] = {}
        self.phase_change_count: Dict[str, int] = {}

        # Spaces will be initialized after SUMO connection
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None

        # For rendering
        self._sumo_started = False

    def _start_sumo(self) -> None:
        """Start SUMO simulation."""
        if self._sumo_started:
            return

        # Determine SUMO command
        if self.sumocfg_file and Path(self.sumocfg_file).exists():
            sumo_cmd = [
                self.sumo_binary,
                "-c", self.sumocfg_file,
                "--step-length", str(self.config.simulation.step_length),
                "--seed", str(self.config.simulation.seed),
                "--no-warnings",
                "--time-to-teleport", "-1",
            ]
        elif self.net_file and self.route_file:
            sumo_cmd = [
                self.sumo_binary,
                "-n", self.net_file,
                "-r", self.route_file,
                "--step-length", str(self.config.simulation.step_length),
                "--seed", str(self.config.simulation.seed),
                "--no-warnings",
                "--time-to-teleport", "-1",
            ]
        else:
            raise ValueError("Either sumocfg_file or both net_file and route_file must be provided")

        if "SUMO_HOME" not in os.environ:
            raise RuntimeError("SUMO_HOME environment variable not set")

        # Start SUMO
        traci.start(sumo_cmd)
        self.conn = traci
        self._sumo_started = True

        # Initialize traffic light and lane info
        self._init_tls_info()
        self._init_spaces()

    def _init_tls_info(self) -> None:
        """Initialize traffic light and incoming lane information."""
        self.tls_ids = self.conn.trafficlight.getIDList()

        for tls_id in self.tls_ids:
            # Get controlled lanes
            controlled_lanes = self.conn.trafficlight.getControlledLanes(tls_id)
            # Get unique incoming lanes (remove duplicates)
            incoming = list(dict.fromkeys(controlled_lanes))
            self.incoming_lanes[tls_id] = incoming

            # Get phase definitions
            program = self.conn.trafficlight.getAllProgramLogics(tls_id)[0]
            self.phase_definitions[tls_id] = [
                {"state": p.state, "duration": p.duration, "minDur": p.minDur, "maxDur": p.maxDur}
                for p in program.phases
            ]

            # Initialize tracking
            self.last_phase[tls_id] = 0
            self.phase_change_count[tls_id] = 0

    def _init_spaces(self) -> None:
        """Initialize observation and action spaces."""
        if not self.tls_ids:
            raise RuntimeError("No traffic lights found in network")

        # For multi-intersection, we concatenate observations
        # For single intersection, we use the first TLS
        primary_tls = self.tls_ids[0]
        num_lanes = len(self.incoming_lanes[primary_tls])
        num_phases = len(self.phase_definitions[primary_tls])

        # Observation: [queue_lengths (num_lanes), wait_times (num_lanes), phase_onehot (num_phases)]
        obs_dim = num_lanes * 2 + num_phases
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: phase duration for each phase (10-60 seconds)
        # We'll use a single action for the primary intersection
        self.action_space = spaces.Box(
            low=10.0,
            high=60.0,
            shape=(num_phases,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if self._sumo_started:
            self.conn.close()
            self._sumo_started = False

        self.current_step = 0
        self.last_phase = {tls: 0 for tls in self.tls_ids}
        self.phase_change_count = {tls: 0 for tls in self.tls_ids}

        self._start_sumo()

        # Run a few steps to let vehicles enter the network
        for _ in range(10):
            self.conn.simulationStep()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if not self._sumo_started:
            raise RuntimeError("Environment not started. Call reset() first.")

        # Apply action: set phase durations
        primary_tls = self.tls_ids[0]
        phase_durations = np.clip(action, 10, 60).astype(int)

        # Get current phase
        current_phase = self.conn.trafficlight.getPhase(primary_tls)

        # Check if phase changed
        if current_phase != self.last_phase[primary_tls]:
            self.phase_change_count[primary_tls] += 1
            self.last_phase[primary_tls] = current_phase

        # Set phase durations
        logic = self.conn.trafficlight.getAllProgramLogics(primary_tls)[0]
        for i, phase in enumerate(logic.phases):
            phase.duration = phase_durations[i]
            phase.minDur = phase_durations[i]
            phase.maxDur = phase_durations[i]
        self.conn.trafficlight.setProgramLogic(primary_tls, logic)

        # Step simulation
        self.conn.simulationStep()
        self.current_step += 1

        # Get observation, reward, done
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self.current_step >= self.total_steps
        truncated = self.conn.simulation.getMinExpectedNumber() <= 0

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        primary_tls = self.tls_ids[0]
        lanes = self.incoming_lanes[primary_tls]
        num_phases = len(self.phase_definitions[primary_tls])

        # Queue lengths
        queue_lengths = np.array([
            self.conn.lane.getLastStepHaltingNumber(lane) for lane in lanes
        ], dtype=np.float32)

        # Wait times (accumulated waiting time)
        wait_times = np.array([
            self.conn.lane.getWaitingTime(lane) for lane in lanes
        ], dtype=np.float32)

        # Current phase (one-hot)
        current_phase = self.conn.trafficlight.getPhase(primary_tls)
        phase_onehot = np.zeros(num_phases, dtype=np.float32)
        if current_phase < num_phases:
            phase_onehot[current_phase] = 1.0

        obs = np.concatenate([queue_lengths, wait_times, phase_onehot])
        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        """Compute reward based on waiting times and phase changes."""
        primary_tls = self.tls_ids[0]
        lanes = self.incoming_lanes[primary_tls]

        # Sum of waiting times across all incoming lanes
        total_wait_time = sum(self.conn.lane.getWaitingTime(lane) for lane in lanes)

        # Phase change penalty
        phase_changes = self.phase_change_count[primary_tls]

        reward = (
            self.config.reward.wait_time_weight * total_wait_time +
            self.config.reward.phase_change_penalty * phase_changes
        )

        return float(reward)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        primary_tls = self.tls_ids[0]
        lanes = self.incoming_lanes[primary_tls]

        total_wait = sum(self.conn.lane.getWaitingTime(lane) for lane in lanes)
        total_queue = sum(self.conn.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        current_phase = self.conn.trafficlight.getPhase(primary_tls)

        return {
            "step": self.current_step,
            "total_wait_time": total_wait,
            "total_queue_length": total_queue,
            "current_phase": current_phase,
            "phase_changes": self.phase_change_count[primary_tls],
            "tls_id": primary_tls,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            # SUMO GUI handles rendering
            return None
        elif self.render_mode == "rgb_array":
            # Would need to implement screenshot capture
            return None
        return None

    def close(self) -> None:
        """Close the environment."""
        if self._sumo_started and self.conn:
            self.conn.close()
            self._sumo_started = False

    def __del__(self) -> None:
        self.close()


def make_env(
    config_path: Optional[str] = None,
    config: Optional[Config] = None,
    **kwargs,
) -> TrafficEnv:
    """Factory function to create TrafficEnv."""
    return TrafficEnv(config_path=config_path, config=config, **kwargs)