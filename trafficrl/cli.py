"""Command-line interface for TrafficRL."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from trafficrl.config import Config, load_config, save_config
from trafficrl.network import generate_network, create_sumo_config
from trafficrl.train import train, train_single_stage
from trafficrl.evaluate import (
    evaluate_model_file,
    compare_models,
    save_results,
    print_results,
)
from trafficrl.env import make_env


def generate_config(args: argparse.Namespace) -> None:
    """Generate default configuration file."""
    config = Config()
    config.network.type = args.network_type
    if args.network_type == "grid":
        config.network.grid_size = args.grid_size

    output_path = args.output or f"configs/{args.network_type}.yaml"
    save_config(config, output_path)
    print(f"Generated config: {output_path}")


def generate_network_cmd(args: argparse.Namespace) -> None:
    """Generate SUMO network file."""
    output_path = args.output or f"networks/{args.network_type}.net.xml"

    if args.network_type == "grid":
        generate_network(
            output_path,
            network_type="grid",
            grid_size=args.grid_size,
            lane_length=args.lane_length,
            num_lanes=args.num_lanes,
            speed_limit=args.speed_limit,
        )
    else:
        generate_network(
            output_path,
            network_type=args.network_type,
            lane_length=args.lane_length,
            num_lanes=args.num_lanes,
            speed_limit=args.speed_limit,
        )

    print(f"Generated network: {output_path}")


def create_routes(args: argparse.Namespace) -> None:
    """Create route file using randomTrips.py."""
    import subprocess

    net_file = args.net_file or f"networks/{args.network_type}.net.xml"
    route_file = args.output or f"routes/{args.network_type}.rou.xml"

    Path(route_file).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "randomTrips.py",
        "-n", net_file,
        "-r", route_file,
        "-e", str(args.steps),
        "--seed", str(args.seed),
        "--prefix", "veh",
        "--min-distance", "100",
        "--max-distance", "5000",
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Generated routes: {route_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating routes: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("randomTrips.py not found. Ensure SUMO tools are in PATH.")
        sys.exit(1)


def train_cmd(args: argparse.Namespace) -> None:
    """Train PPO agent."""
    config = load_config(args.config)

    # Override config with CLI args
    if args.timesteps:
        config.agent.total_timesteps = args.timesteps
    if args.seed:
        config.simulation.seed = args.seed
    if args.n_envs:
        # Would need to modify train function to accept n_envs
        pass

    print(f"Training with config: {args.config}")
    print(f"Network: {config.network.type}")
    print(f"Timesteps: {config.agent.total_timesteps}")

    model = train(config, seed=config.simulation.seed)

    # Save final model
    save_path = args.model or config.model_save_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Model saved to: {save_path}")


def evaluate_cmd(args: argparse.Namespace) -> None:
    """Evaluate trained model."""
    config = load_config(args.config)

    model_path = args.model or config.model_save_path
    vecnormalize_path = args.vecnormalize

    results = evaluate_model_file(
        model_path,
        config,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render,
        vecnormalize_path=vecnormalize_path,
    )

    print_results(results)

    if args.output:
        save_results(results, args.output)
        print(f"Results saved to: {args.output}")


def compare_cmd(args: argparse.Namespace) -> None:
    """Compare multiple models."""
    config = load_config(args.config)

    model_paths = args.models
    results = compare_models(model_paths, config, n_episodes=args.episodes, deterministic=args.deterministic)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for name, metrics in results["summary"].items():
        print(f"{name:30s} Reward: {metrics['reward_mean']:8.2f} ± {metrics['reward_std']:6.2f}  "
              f"Wait: {metrics['wait_time_mean']:8.1f}  Queue: {metrics['queue_length_mean']:6.1f}")
    print("=" * 60)

    if args.output:
        save_results(results, args.output)
        print(f"Results saved to: {args.output}")


def test_env_cmd(args: argparse.Namespace) -> None:
    """Test environment reset/step."""
    config = load_config(args.config)
    env = make_env(config=config)

    print("Testing environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")

    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 100 == 0:
            print(f"Step {i}: reward={reward:.2f}, wait={info.get('total_wait_time', 0):.1f}")
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break

    env.close()
    print("Environment test passed!")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TrafficRL: SUMO traffic simulation + PPO agent for adaptive signal control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-config
    gen_config = subparsers.add_parser("generate-config", help="Generate default config file")
    gen_config.add_argument("--network-type", choices=["single", "double", "grid"], default="single")
    gen_config.add_argument("--grid-size", type=int, default=2)
    gen_config.add_argument("--output", "-o", help="Output config path")
    gen_config.set_defaults(func=generate_config)

    # generate-network
    gen_net = subparsers.add_parser("generate-network", help="Generate SUMO network file")
    gen_net.add_argument("--network-type", choices=["single", "double", "grid"], default="single")
    gen_net.add_argument("--grid-size", type=int, default=2)
    gen_net.add_argument("--lane-length", type=float, default=100.0)
    gen_net.add_argument("--num-lanes", type=int, default=1)
    gen_net.add_argument("--speed-limit", type=float, default=13.89)
    gen_net.add_argument("--output", "-o", help="Output network path")
    gen_net.set_defaults(func=generate_network_cmd)

    # create-routes
    gen_routes = subparsers.add_parser("create-routes", help="Create route file using randomTrips.py")
    gen_routes.add_argument("--network-type", choices=["single", "double", "grid"], default="single")
    gen_routes.add_argument("--net-file", help="Network file path")
    gen_routes.add_argument("--steps", type=int, default=3600)
    gen_routes.add_argument("--seed", type=int, default=42)
    gen_routes.add_argument("--output", "-o", help="Output route file path")
    gen_routes.set_defaults(func=create_routes)

    # train
    train_parser = subparsers.add_parser("train", help="Train PPO agent")
    train_parser.add_argument("--config", "-c", required=True, help="Config file path")
    train_parser.add_argument("--timesteps", type=int, help="Override total timesteps")
    train_parser.add_argument("--seed", type=int, help="Random seed")
    train_parser.add_argument("--n-envs", type=int, help="Number of parallel environments")
    train_parser.add_argument("--model", "-m", help="Output model path")
    train_parser.set_defaults(func=train_cmd)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--config", "-c", required=True, help="Config file path")
    eval_parser.add_argument("--model", "-m", help="Model file path")
    eval_parser.add_argument("--vecnormalize", help="VecNormalize stats path")
    eval_parser.add_argument("--episodes", "-n", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--deterministic", action="store_true", default=True)
    eval_parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    eval_parser.add_argument("--render", action="store_true", help="Render episodes")
    eval_parser.add_argument("--output", "-o", help="Output results JSON path")
    eval_parser.set_defaults(func=evaluate_cmd)

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--config", "-c", required=True, help="Config file path")
    compare_parser.add_argument("--models", "-m", nargs="+", required=True, help="Model file paths")
    compare_parser.add_argument("--episodes", "-n", type=int, default=10, help="Episodes per model")
    compare_parser.add_argument("--deterministic", action="store_true", default=True)
    compare_parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    compare_parser.add_argument("--output", "-o", help="Output results JSON path")
    compare_parser.set_defaults(func=compare_cmd)

    # test-env
    test_parser = subparsers.add_parser("test-env", help="Test environment reset/step")
    test_parser.add_argument("--config", "-c", required=True, help="Config file path")
    test_parser.add_argument("--steps", type=int, default=100, help="Number of steps to test")
    test_parser.set_defaults(func=test_env_cmd)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()