"""SUMO network generation for TrafficRL environments."""

import os
import subprocess
from pathlib import Path
from typing import Optional


def generate_network(
    output_path: str,
    network_type: str = "single",
    grid_size: int = 2,
    lane_length: float = 100.0,
    num_lanes: int = 1,
    speed_limit: float = 13.89,
) -> str:
    """
    Generate a SUMO network file using netgenerate.

    Args:
        output_path: Path to save the .net.xml file
        network_type: One of 'single', 'double', 'grid'
        grid_size: For grid networks, number of intersections per side
        lane_length: Length of each lane in meters
        num_lanes: Number of lanes per edge
        speed_limit: Speed limit in m/s (default 50 km/h = 13.89 m/s)

    Returns:
        Path to the generated network file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if network_type == "single":
        return _generate_single_intersection(output_path, lane_length, num_lanes, speed_limit)
    elif network_type == "double":
        return _generate_double_intersection(output_path, lane_length, num_lanes, speed_limit)
    elif network_type == "grid":
        return _generate_grid_network(output_path, grid_size, lane_length, num_lanes, speed_limit)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def _generate_single_intersection(
    output_path: Path,
    lane_length: float,
    num_lanes: int,
    speed_limit: float,
) -> str:
    """Generate a single 4-way intersection."""
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.x-number", "2",
        "--grid.y-number", "2",
        "--grid.length", str(lane_length),
        "--grid.attach-length", str(lane_length),
        "--default.speed", str(speed_limit),
        "--output-file", str(output_path),
        "--no-turnarounds",
        "--tls.default-type", "actuated",
    ]
    _run_netgenerate(cmd)
    return str(output_path)


def _generate_double_intersection(
    output_path: Path,
    lane_length: float,
    num_lanes: int,
    speed_limit: float,
) -> str:
    """Generate two connected intersections."""
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.x-number", "3",
        "--grid.y-number", "2",
        "--grid.length", str(lane_length),
        "--grid.attach-length", str(lane_length),
        "--default.speed", str(speed_limit),
        "--output-file", str(output_path),
        "--no-turnarounds",
        "--tls.default-type", "actuated",
    ]
    _run_netgenerate(cmd)
    return str(output_path)


def _generate_grid_network(
    output_path: Path,
    grid_size: int,
    lane_length: float,
    num_lanes: int,
    speed_limit: float,
) -> str:
    """Generate a grid of intersections."""
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.x-number", str(grid_size + 1),
        "--grid.y-number", str(grid_size + 1),
        "--grid.length", str(lane_length),
        "--grid.attach-length", str(lane_length),
        "--default.speed", str(speed_limit),
        "--output-file", str(output_path),
        "--no-turnarounds",
        "--tls.default-type", "actuated",
    ]
    _run_netgenerate(cmd)
    return str(output_path)


def _run_netgenerate(cmd: list[str]) -> None:
    """Run netgenerate command and handle errors."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stderr:
            print(f"netgenerate stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"netgenerate failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("netgenerate not found. Ensure SUMO is installed and in PATH.")


def create_sumo_config(
    network_file: str,
    route_file: str,
    output_path: str,
    simulation_steps: int = 3600,
    begin: int = 0,
    end: int = 3600,
) -> str:
    """
    Create a SUMO configuration file (.sumocfg).

    Args:
        network_file: Path to .net.xml file
        route_file: Path to .rou.xml file
        output_path: Path to save .sumocfg file
        simulation_steps: Total simulation steps
        begin: Simulation start time
        end: Simulation end time

    Returns:
        Path to the generated config file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_content = f"""<configuration>
    <input>
        <net-file value="{network_file}"/>
        <route-files value="{route_file}"/>
    </input>
    <time>
        <begin value="{begin}"/>
        <end value="{end}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="-1"/>
    </processing>
    <output>
        <tripinfo-output value="tripinfo.xml"/>
        <summary-output value="summary.xml"/>
    </output>
</configuration>"""

    output_path.write_text(config_content)
    return str(output_path)