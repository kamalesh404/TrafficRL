"""Evaluation module for TrafficRL trained agents."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from trafficrl.config import Config
from trafficrl.env import TrafficEnv, make_env


def evaluate_model(
    model: PPO,
    config: Config,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    vecnormalize_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained PPO model.

    Args:
        model: Trained PPO model
        config: Configuration object
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        render: Whether to render episodes
        vecnormalize_path: Path to VecNormalize stats (for normalized envs)

    Returns:
        Dictionary with evaluation metrics
    """
    # Create environment
    env = make_env(config=config)
    if render:
        env.render_mode = "human"

    # Load VecNormalize if provided
    if vecnormalize_path and Path(vecnormalize_path).exists():
        # We need to wrap in VecNormalize for the model to work correctly
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        def make_eval_env():
            return Monitor(make_env(config=config))

        vec_env = DummyVecEnv([make_eval_env])
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        eval_env = vec_env
    else:
        eval_env = env

    episode_rewards = []
    episode_lengths = []
    episode_wait_times = []
    episode_queue_lengths = []
    episode_phase_changes = []

    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0
        total_wait = 0.0
        total_queue = 0.0
        phase_changes = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = eval_env.step(action)

            episode_reward += reward
            episode_length += 1

            if isinstance(info, list):
                info = info[0]

            total_wait += info.get("total_wait_time", 0)
            total_queue += info.get("total_queue_length", 0)
            phase_changes = info.get("phase_changes", 0)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_wait_times.append(total_wait)
        episode_queue_lengths.append(total_queue)
        episode_phase_changes.append(phase_changes)

    eval_env.close()

    # Compute statistics
    results = {
        "n_episodes": n_episodes,
        "reward_mean": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "reward_min": float(np.min(episode_rewards)),
        "reward_max": float(np.max(episode_rewards)),
        "length_mean": float(np.mean(episode_lengths)),
        "length_std": float(np.std(episode_lengths)),
        "wait_time_mean": float(np.mean(episode_wait_times)),
        "wait_time_std": float(np.std(episode_wait_times)),
        "queue_length_mean": float(np.mean(episode_queue_lengths)),
        "queue_length_std": float(np.std(episode_queue_lengths)),
        "phase_changes_mean": float(np.mean(episode_phase_changes)),
        "phase_changes_std": float(np.std(episode_phase_changes)),
        "episodes": [
            {
                "episode": i,
                "reward": float(episode_rewards[i]),
                "length": int(episode_lengths[i]),
                "wait_time": float(episode_wait_times[i]),
                "queue_length": float(episode_queue_lengths[i]),
                "phase_changes": int(episode_phase_changes[i]),
            }
            for i in range(n_episodes)
        ],
    }

    return results


def evaluate_model_file(
    model_path: str,
    config: Config,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    vecnormalize_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model from file.

    Args:
        model_path: Path to saved model (.zip)
        config: Configuration object
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        render: Whether to render episodes
        vecnormalize_path: Path to VecNormalize stats

    Returns:
        Dictionary with evaluation metrics
    """
    model = PPO.load(model_path)
    return evaluate_model(
        model,
        config,
        n_episodes=n_episodes,
        deterministic=deterministic,
        render=render,
        vecnormalize_path=vecnormalize_path,
    )


def compare_models(
    model_paths: List[str],
    config: Config,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Compare multiple models.

    Args:
        model_paths: List of model paths
        config: Configuration object
        n_episodes: Number of evaluation episodes per model
        deterministic: Whether to use deterministic actions

    Returns:
        Dictionary with comparison results
    """
    results = {}
    for path in model_paths:
        name = Path(path).stem
        print(f"Evaluating {name}...")
        results[name] = evaluate_model_file(path, config, n_episodes, deterministic)

    # Summary
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "reward_mean": res["reward_mean"],
            "reward_std": res["reward_std"],
            "wait_time_mean": res["wait_time_mean"],
            "queue_length_mean": res["queue_length_mean"],
        }

    return {
        "detailed": results,
        "summary": summary,
    }


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def print_results(results: Dict[str, Any]) -> None:
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {results['n_episodes']}")
    print(f"Reward:   {results['reward_mean']:.2f} ± {results['reward_std']:.2f}")
    print(f"Length:   {results['length_mean']:.1f} ± {results['length_std']:.1f}")
    print(f"Wait Time: {results['wait_time_mean']:.1f} ± {results['wait_time_std']:.1f}")
    print(f"Queue Len: {results['queue_length_mean']:.1f} ± {results['queue_length_std']:.1f}")
    print(f"Phase Chg: {results['phase_changes_mean']:.1f} ± {results['phase_changes_std']:.1f}")
    print("=" * 50)