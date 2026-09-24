"""Training module for TrafficRL PPO agent."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from trafficrl.config import Config, AgentConfig, CurriculumConfig
from trafficrl.env import TrafficEnv, make_env


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            if "total_wait_time" in info:
                self.logger.record("env/total_wait_time", info["total_wait_time"])
            if "total_queue_length" in info:
                self.logger.record("env/total_queue_length", info["total_queue_length"])
            if "phase_changes" in info:
                self.logger.record("env/phase_changes", info["phase_changes"])
        return True


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning - switches environment configuration."""

    def __init__(
        self,
        curriculum_config: CurriculumConfig,
        env_fn,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.curriculum_config = curriculum_config
        self.env_fn = env_fn
        self.current_stage = 0
        self.stage_timesteps = curriculum_config.stage_timesteps
        self.stages = curriculum_config.stages

    def _on_step(self) -> bool:
        if not self.curriculum_config.enabled:
            return True

        current_timesteps = self.num_timesteps

        # Check if we should advance to next stage
        while (
            self.current_stage < len(self.stage_timesteps) - 1
            and current_timesteps >= sum(self.stage_timesteps[: self.current_stage + 1])
        ):
            self.current_stage += 1
            self.logger.record("curriculum/stage", self.current_stage)
            self.logger.record("curriculum/stage_name", self.stages[self.current_stage])
            if self.verbose > 0:
                print(f"\nCurriculum: Advancing to stage {self.current_stage} ({self.stages[self.current_stage]})")

        return True


def create_env(config: Config, rank: int = 0, seed: int = 0) -> TrafficEnv:
    """Create a single environment instance."""
    env = make_env(config=config)
    env.reset(seed=seed + rank)
    return Monitor(env)


def create_vec_env(
    config: Config,
    n_envs: int = 1,
    seed: int = 0,
    vec_env_cls: str = "dummy",
) -> Union[DummyVecEnv, SubprocVecEnv]:
    """Create vectorized environment."""
    env_fns = [lambda r=i: create_env(config, r, seed) for i in range(n_envs)]

    if vec_env_cls == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def train_single_stage(
    config: Config,
    stage_name: str,
    timesteps: int,
    model: Optional[PPO] = None,
    tensorboard_log: Optional[str] = None,
    n_envs: int = 1,
    seed: int = 42,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
    n_eval_episodes: int = 5,
    vec_env_cls: str = "dummy",
) -> PPO:
    """
    Train PPO agent for a single curriculum stage.

    Args:
        config: Configuration object
        stage_name: Name of current stage (for logging)
        timesteps: Number of timesteps to train
        model: Existing model to continue training (None for new)
        tensorboard_log: TensorBoard log directory
        n_envs: Number of parallel environments
        seed: Random seed
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        n_eval_episodes: Episodes per evaluation
        vec_env_cls: Vector env class ('dummy' or 'subproc')

    Returns:
        Trained PPO model
    """
    # Create vectorized environment
    vec_env = create_vec_env(config, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Setup logging
    log_dir = Path(tensorboard_log or config.agent.tensorboard_log) / stage_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_{stage_name}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_env = create_vec_env(config, n_envs=1, seed=seed + 1000, vec_env_cls="dummy")
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # TensorBoard callback
    callbacks.append(TensorboardCallback())

    callback_list = CallbackList(callbacks)

    # Create or load model
    if model is None:
        model = PPO(
            policy=config.agent.policy,
            env=vec_env,
            learning_rate=config.agent.learning_rate,
            n_steps=config.agent.n_steps,
            batch_size=config.agent.batch_size,
            n_epochs=config.agent.n_epochs,
            gamma=config.agent.gamma,
            gae_lambda=config.agent.gae_lambda,
            clip_range=config.agent.clip_range,
            ent_coef=config.agent.ent_coef,
            vf_coef=config.agent.vf_coef,
            max_grad_norm=config.agent.max_grad_norm,
            tensorboard_log=str(log_dir),
            verbose=1,
            seed=seed,
            device="auto",
        )
    else:
        model.set_env(vec_env)

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=callback_list,
        tb_log_name=stage_name,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    # Save final model
    model.save(str(log_dir / f"ppo_{stage_name}_final"))

    # Save VecNormalize stats
    vec_env.save(str(log_dir / "vecnormalize.pkl"))

    return model


def train_curriculum(config: Config, seed: int = 42) -> PPO:
    """
    Train agent through curriculum stages.

    Args:
        config: Configuration object
        seed: Random seed

    Returns:
        Final trained model
    """
    curriculum = config.curriculum
    model = None

    for i, stage in enumerate(curriculum.stages):
        timesteps = curriculum.stage_timesteps[i]

        # Update config for this stage
        stage_config = Config(
            network=Config(
                network=config.network,
                simulation=config.simulation,
                agent=config.agent,
                reward=config.reward,
                eval=config.eval,
                curriculum=config.curriculum,
            ).network
        )
        stage_config.network.type = stage

        if stage == "grid":
            stage_config.network.grid_size = config.network.grid_size

        print(f"\n{'='*50}")
        print(f"Curriculum Stage {i+1}/{len(curriculum.stages)}: {stage}")
        print(f"Timesteps: {timesteps}")
        print(f"{'='*50}")

        model = train_single_stage(
            config=stage_config,
            stage_name=stage,
            timesteps=timesteps,
            model=model,
            tensorboard_log=config.agent.tensorboard_log,
            n_envs=1,
            seed=seed + i * 100,
        )

    return model


def train(config: Config, seed: int = 42) -> PPO:
    """
    Main training entry point.

    Args:
        config: Configuration object
        seed: Random seed

    Returns:
        Trained PPO model
    """
    if config.curriculum.enabled and len(config.curriculum.stages) > 1:
        return train_curriculum(config, seed)
    else:
        return train_single_stage(
            config=config,
            stage_name=config.network.type,
            timesteps=config.agent.total_timesteps,
            tensorboard_log=config.agent.tensorboard_log,
            seed=seed,
        )