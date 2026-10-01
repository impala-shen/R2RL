"""Task-selection checks that never connect to a robot."""

import argparse
import contextlib
import io
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from task_settings import ROOT, TASKS, add_task_arguments, create_robot_env, resolve_demo, validate_demo


class TaskSelectionTests(unittest.TestCase):
    def test_included_demonstrations_match_task_dimensions(self):
        for name in ("snapfit", "peginhole", "assembly"):
            task = TASKS[name]
            with self.subTest(task=name):
                self.assertGreater(validate_demo(resolve_demo(task, None), task), 0)
                self.assertGreater(validate_demo(resolve_demo(task, None, ibrl=True), task), 0)

    def test_mismatched_task_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 18 observations"):
            validate_demo(TASKS["snapfit"].demo_file, TASKS["peginhole"])

    def test_openbox_requires_its_own_demonstration(self):
        with self.assertRaisesRegex(ValueError, "supply --demo-file"):
            validate_demo(resolve_demo(TASKS["openbox"], None), TASKS["openbox"])

    def test_empty_demonstration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            header = TASKS["snapfit"].demo_file.read_text().splitlines()[0]
            path.write_text(header + "\n")
            with self.assertRaisesRegex(ValueError, "no transitions"):
                validate_demo(path, TASKS["snapfit"])

    def test_task_options_and_overrides(self):
        parser = argparse.ArgumentParser()
        add_task_arguments(parser, default_task="snapfit")
        args = parser.parse_args([
            "--task", "assembly", "--robot-ip", "127.0.0.1",
            "--home-pose", "1", "2", "3", "0", "0", "0", "--max-steps", "20",
        ])
        self.assertEqual(args.task, "assembly")
        self.assertEqual(args.robot_ip, "127.0.0.1")
        env = create_robot_env(TASKS[args.task], None, None, device="cpu",
                               home_pose=args.home_pose, max_steps=args.max_steps)
        self.assertEqual(env.home.tolist(), [1, 2, 3, 0, 0, 0])
        self.assertEqual(env._MAX_STEPS, 20)

    def test_all_online_environments_construct_without_csv_or_hardware(self):
        with patch("pandas.read_csv", side_effect=AssertionError("online CSV read")):
            for name, task in TASKS.items():
                with self.subTest(task=name):
                    env = create_robot_env(task, None, None, device="cpu")
                    self.assertEqual(type(env).__name__, task.env_class)
                    self.assertEqual(env.observation_space.shape, (task.observation_dim,))
                    self.assertEqual(env.action_space.shape, (6,))
                    self.assertEqual(tuple(env.home), task.home_pose)
                    self.assertFalse(env._offline_mode)

    def test_recorded_observations_use_included_data(self):
        from envs.snapfit_env import SnapFitRobotEnv
        import numpy as np

        env = SnapFitRobotEnv(None, None, np.array(TASKS["snapfit"].home_pose), offline_mode=True)
        observation = env._get_obs()
        self.assertEqual(observation.shape, (19,))
        self.assertTrue(np.isfinite(observation).all())

    def test_invalid_configuration_exits_before_robot_connection(self):
        cases = [
            ("task_R2RL.py", ["--task", "openbox"]),
            ("task_R2RL.py", ["--eval"]),
            ("task_R2RL.py", ["--max-steps", "0"]),
            ("task_IBRL.py", ["--bc-checkpoint", "/missing/bc.pt"]),
            ("task_ResidualRL.py", ["--task", "openbox"]),
            ("task_ResidualRL.py", ["--timesteps", "0"]),
        ]
        for script, arguments in cases:
            with self.subTest(script=script, arguments=arguments):
                with patch("sys.argv", [script, *arguments]), \
                     patch("rtde_control.RTDEControlInterface") as control, \
                     patch("rtde_receive.RTDEReceiveInterface") as receive, \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        runpy.run_path(str(ROOT / script), run_name="__main__")
                    self.assertEqual(error.exception.code, 2)
                    control.assert_not_called()
                    receive.assert_not_called()

    def test_entrypoints_wire_selected_task_into_trainer(self):
        """Build real agents with fake robot interfaces and a non-running trainer."""
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bc.pt"
            imitation = nn.Sequential(
                nn.Linear(18, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6), nn.Tanh(),
            )
            torch.save({"policy": {f"net.{key}": value for key, value in imitation.state_dict().items()}}, checkpoint)
            cases = [
                ("task_R2RL.py", "assembly", []),
                ("task_IBRL.py", "peginhole", ["--bc-checkpoint", str(checkpoint)]),
                ("task_ResidualRL.py", "snapfit", []),
            ]
            for script, name, arguments in cases:
                with self.subTest(script=script, task=name):
                    with patch("sys.argv", [script, "--task", name, "--robot-ip", "127.0.0.1", *arguments]), \
                         patch("rtde_control.RTDEControlInterface") as control, \
                         patch("rtde_receive.RTDEReceiveInterface") as receive, \
                         patch("torch.cuda.is_available", return_value=False), \
                         patch("skrl.trainers.torch.SequentialTrainer") as trainer:
                        namespace = runpy.run_path(str(ROOT / script), run_name="__main__")
                        control.assert_called_once_with("127.0.0.1")
                        receive.assert_called_once_with("127.0.0.1")
                        trainer.return_value.train.assert_called_once()
                        env = namespace["raw_env"]
                        self.assertEqual(type(env).__name__, TASKS[name].env_class)
                        self.assertEqual(tuple(env.home), TASKS[name].home_pose)
                        self.assertEqual(namespace["cfg"].demo_file, str(resolve_demo(TASKS[name], None, ibrl=script == "task_IBRL.py")))


if __name__ == "__main__":
    unittest.main()
