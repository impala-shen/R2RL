"""Shared task selection for the real-robot training entry points.

Home poses are copied from the existing task scripts and are fixture-specific.
No robot connection is created by importing this module.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMONSTRATIONS = ROOT / "data" / "demonstrations"
DEFAULT_ROBOT_IP = "192.168.1.11"


@dataclass(frozen=True)
class TaskSettings:
    env_module: str
    env_class: str
    home_pose: tuple[float, ...]
    observation_dim: int
    demo_file: Path | None = None
    ibrl_demo_file: Path | None = None
    max_steps: int = 100
    requires_gripper: bool = False


TASKS = {
    "snapfit": TaskSettings(
        "envs.snapfit_env", "SnapFitRobotEnv",
        (-0.35837264, -0.07759438, 0.23013212, -3.04344862, -0.60611295, -0.16027859),
        19,
        DEMONSTRATIONS / "SnapFit" / "snapfit_fixscale.csv",
        DEMONSTRATIONS / "SnapFit" / "10demos.csv",
    ),
    "peginhole": TaskSettings(
        "envs.peginhole", "PegInHoleRobotEnv",
        (-0.38633344, -0.22858439, 0.10802976, -3.06064544, -0.6274192, 0.03773944),
        18,
        DEMONSTRATIONS / "PegInHole" / "10demos.csv",
        DEMONSTRATIONS / "PegInHole" / "10demos.csv",
        requires_gripper=True,
    ),
    "assembly": TaskSettings(
        "envs.plugin", "AssemblyRobotEnv",
        (-0.39687852, 0.01385717, 0.14660621, 0.99239598, 1.41831132, -0.89998169),
        18,
        DEMONSTRATIONS / "PegInHole" / "assembly.csv",
        DEMONSTRATIONS / "PegInHole" / "assembly.csv",
        requires_gripper=True,
    ),
    "openbox": TaskSettings(
        "envs.open_box", "OpenBoxRobotEnv",
        (-0.50922925, -0.12718483, 0.01599379, 2.96554952, 0.67491036, 0.05593863),
        19,
        # No open-box demonstration is included. Supply --demo-file explicitly.
    ),
}


def add_task_arguments(parser: argparse.ArgumentParser, *, default_task: str) -> None:
    parser.add_argument("--task", choices=tuple(TASKS), default=default_task)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--demo-file", default=None, help="Override the selected task's demonstration CSV")
    parser.add_argument("--home-pose", type=float, nargs=6, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="Override the task home pose (meters and rotation-vector radians)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override the task's episode step limit")


def resolve_demo(task: TaskSettings, override: str | None, *, ibrl: bool = False) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    return task.ibrl_demo_file if ibrl else task.demo_file


def validate_demo(path: Path | None, task: TaskSettings) -> int:
    """Check task dimensions and count transitions before connecting hardware."""
    if path is None:
        raise ValueError("No demonstration is configured for this task; supply --demo-file")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            *(f"observations.{i}" for i in range(task.observation_dim)),
            *(f"next_observations.{i}" for i in range(task.observation_dim)),
            *(f"actions.{i}" for i in range(6)),
            "rewards.0", "terminated.0", "truncated.0",
        }
        columns = set(reader.fieldnames or ())
        missing = required - columns
        for group, size in (("observations", task.observation_dim),
                            ("next_observations", task.observation_dim), ("actions", 6)):
            if sum(name.startswith(group + ".") for name in columns) != size:
                raise ValueError(f"{path}: expected {size} {group} columns for this task")
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        count = sum(1 for _ in reader)
    if not count:
        raise ValueError(f"{path}: demonstration contains no transitions")
    return count


def create_robot_env(task: TaskSettings, rtde_c, rtde_r, *, device, gripper=None,
                     home_pose=None, max_steps=None):
    """Construct the selected online environment without loading a replay CSV."""
    import importlib
    import numpy as np

    env_class = getattr(importlib.import_module(task.env_module), task.env_class)
    options = {
        "home_pose": np.array(task.home_pose if home_pose is None else home_pose, dtype=np.float64),
        "max_steps": task.max_steps if max_steps is None else max_steps,
        "offline_mode": False,
        "device": device,
    }
    if options["max_steps"] < 1:
        raise ValueError("max_steps must be positive")
    if task.requires_gripper:
        options["gripper"] = gripper
    return env_class(rtde_c, rtde_r, **options)
