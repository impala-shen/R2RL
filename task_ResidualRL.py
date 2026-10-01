"""Train a composable residual policy on top of a fixed base controller.

One-demo example (the default configuration below):
    python task_ResidualRL.py --task snapfit

Checkpoint-policy example:
    python task_ResidualRL.py \
        --base-policy-type checkpoint \
        --base-checkpoint runs/torch/snapfit/BC1/checkpoints/agent_30000.pt

Examples:
    # Deterministic Bellman/maxQ updates
    python task_ResidualRL.py --task snapfit --deterministic-policy --q-method bellman --policy-method maxQ

    # Stochastic Bellman critic with DDPG+BC actor regularization
    python task_ResidualRL.py --task snapfit --no-deterministic-policy \
        --q-method bellman --policy-method DDPG+BC

    # Deterministic IQL critic target with DDPG+BC actor regularization
    python task_ResidualRL.py --task snapfit --deterministic-policy --q-method iql --policy-method DDPG+BC
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from skrl import logger
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from algorithms.ResidualRL import ResidualRL, ResidualRL_CFG
from task_settings import TASKS, add_task_arguments, resolve_demo, validate_demo, create_robot_env
from models.gaussian import GaussianMixin
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq.robotiq_gripper_control import RobotiqGripper

set_seed(42)


class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, *, zero_output=False):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Tanh(),
        )
        if zero_output:
            nn.init.zeros_(self.net[-2].weight)
            nn.init.zeros_(self.net[-2].bias)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}



class StochasticActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-5,
        max_log_std=1,
        reduction="sum",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_layer = nn.Linear(64, self.num_actions)
        # nn.init.zeros_(self.log_std_layer.weight)
        # nn.init.constant_(self.log_std_layer.bias, -2.0)  # std ≈ 0.135

    def compute(self, inputs, role):
        x = self.net(inputs["observations"])

        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)

        return mean, {"log_std": log_std}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        features = torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1)
        return self.net(features), {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}

# Choose the base controller here. CLI arguments can override these values.
DEFAULT_TASK = "openbox"
BASE_POLICY_TYPE = "one_demo"  # "one_demo" or "checkpoint"
DETERMINISTIC_POLICY = False
Q_METHOD = "bellman"  # "bellman" or "iql"
POLICY_METHOD = "maxQ"  # "maxQ" or "DDPG+BC"
DEMO_FILE_PATH = None
BASE_CHECKPOINT_PATH = None

parser = argparse.ArgumentParser()
add_task_arguments(parser, default_task=DEFAULT_TASK)
parser.set_defaults(demo_file=DEMO_FILE_PATH)
parser.add_argument(
    "--deterministic-policy",
    action=argparse.BooleanOptionalAction,
    default=DETERMINISTIC_POLICY,
    help="Use deterministic policy; pass --no-deterministic-policy for a stochastic policy",
)
parser.add_argument(
    "--q-method",
    choices=("bellman", "iql"),
    default=Q_METHOD,
    help="Critic update target",
)
parser.add_argument(
    "--policy-method",
    choices=("maxQ", "DDPG+BC"),
    default=POLICY_METHOD,
    help="Actor update objective",
)
parser.add_argument(
    "--base-policy-type",
    choices=("one_demo", "checkpoint"),
    default=BASE_POLICY_TYPE,
    help="Fixed controller used underneath the learned residual",
)
parser.add_argument(
    "--base-checkpoint",
    default=BASE_CHECKPOINT_PATH,
    help="Frozen base-controller checkpoint (overrides BASE_CHECKPOINT_PATH)",
)
parser.add_argument("--checkpoint", default=None, help="Optional ResidualRL checkpoint")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--timesteps", type=int, default=10000)
args, _ = parser.parse_known_args()


task = TASKS[args.task]
if args.max_steps is not None and args.max_steps < 1:
    parser.error("--max-steps must be positive")
if args.timesteps < 1:
    parser.error("--timesteps must be positive")
if args.checkpoint and not os.path.isfile(args.checkpoint):
    parser.error(f"ResidualRL checkpoint not found: {args.checkpoint}")
demo_rows = 0
if args.base_policy_type == "checkpoint":
    if not args.base_checkpoint or not os.path.isfile(args.base_checkpoint):
        parser.error("Set BASE_CHECKPOINT_PATH or supply an existing --base-checkpoint")
else:
    demo_file = resolve_demo(task, args.demo_file)
    try:
        demo_rows = validate_demo(demo_file, task)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.demo_file = str(demo_file)

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"


# Shared task selection keeps the environment, home pose, and data together.
ROBOT_IP = args.robot_ip
rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)
rtde_c.zeroFtSensor()
gripper_ = RobotiqGripper(rtde_c) if task.requires_gripper else None
MAX_STEPS = args.max_steps if args.max_steps is not None else task.max_steps
task_name = args.task
raw_env = create_robot_env(
    task, rtde_c, rtde_r, gripper=gripper_, device=device_name,
    home_pose=args.home_pose, max_steps=MAX_STEPS,
)
logger.info(f"Task: {task_name}, home pose: {raw_env.home}")

env = wrap_env(raw_env)
device = env.device

models = {}
if args.deterministic_policy:
    models["policy"] = DeterministicActor(
        env.observation_space, env.state_space, env.action_space, device, zero_output=True
    )
else:
    models["policy"] = StochasticActor(
        env.observation_space, env.state_space, env.action_space, device
    )

if args.deterministic_policy and args.q_method == "bellman":
    models["target_policy"] = DeterministicActor(
        env.observation_space, env.state_space, env.action_space, device, zero_output=True
    )

models["critic_1"] = Critic(
    env.observation_space, env.state_space, env.action_space, device
)
models["critic_2"] = Critic(
    env.observation_space, env.state_space, env.action_space, device
)
if args.q_method == "bellman":
    models["target_critic_1"] = Critic(
        env.observation_space, env.state_space, env.action_space, device
    )
    models["target_critic_2"] = Critic(
        env.observation_space, env.state_space, env.action_space, device
    )
if args.q_method == "iql":
    models["value"] = Value(env.observation_space, env.state_space, env.action_space, device)

if args.base_policy_type == "checkpoint":
    base_policy = StochasticActor(
        env.observation_space, env.state_space, env.action_space, device
    )
    base_checkpoint = torch.load(args.base_checkpoint, map_location=device)
    base_policy.load_state_dict(base_checkpoint.get("policy", base_checkpoint))
    base_policy.eval()
    models_ref = {"policy": base_policy}
    expert_memory = None
else:
    models_ref = {}
    demo_length = demo_rows
    expert_memory = RandomMemory(
        memory_size=demo_length,
        num_envs=1,
        device=device,
    )
    logger.info(f"Using one-demo base controller: {args.demo_file} ({demo_length} actions)")

memory = RandomMemory(memory_size=10_000, num_envs=1, device=device)


cfg = ResidualRL_CFG()
cfg.deterministic_policy = args.deterministic_policy
cfg.q_method = args.q_method
cfg.policy_method = args.policy_method
cfg.gradient_steps = 3
cfg.batch_size = 64
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = 3e-4
cfg.policy_delay = 2
cfg.learning_starts = 0
cfg.random_timesteps = 100
cfg.num_envs = 1
cfg.base_policy_type = args.base_policy_type
cfg.demo_file = args.demo_file if args.base_policy_type == "one_demo" else ""
# Start at the base controller (the residual head is zero-initialized), then
# explore only in residual-action space.
cfg.residual_action_scale = 1
if args.deterministic_policy:
    cfg.exploration_noise = GaussianNoise
    cfg.exploration_noise_kwargs = {"mean": 0.0, "std": 0.01, "device": device}
    cfg.exploration_scheduler = lambda timestep, timesteps: max(1 - timestep / timesteps, 0.05)
if args.deterministic_policy and args.q_method == "bellman":
    cfg.smooth_regularization_noise = GaussianNoise
    cfg.smooth_regularization_noise_kwargs = {"mean": 0.0, "std": 0.02, "device": device}
    cfg.smooth_regularization_clip = 0.05

# Stochastic-policy entropy tuning. It is independent of the critic target.
cfg.initial_entropy_value = 0.1
cfg.learn_entropy = False
cfg.target_entropy = None
cfg.entropy_learning_rate = 3e-4

# IQL critic/value setting. The actor objective is still selected independently.
cfg.expectile = 0.7

# DDPG+BC: lambda = ddpg_bc_alpha / mean(abs(Q)); the second coefficient
# directly controls regularization toward residual actions already in replay.
cfg.ddpg_bc_alpha = 1
cfg.bc_loss_scale = 1.0

# The existing BC actors emit normalized actions and were trained on raw observations.
cfg.base_policy_outputs_normalized = True
cfg.preprocess_base_observations = False
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}

cfg.experiment.write_interval = 100 if not args.eval else 0
cfg.experiment.checkpoint_interval = 500 if not args.eval else 0
cfg.experiment.directory = (
    f"runs/torch/{task_name}/residual_rl"
    if args.deterministic_policy and args.q_method == "bellman" and args.policy_method == "maxQ"
    else (
        f"runs/torch/{task_name}/residual_rl/"
        f"{'det' if args.deterministic_policy else 'stoch'}_"
        f"{args.q_method}_{args.policy_method.lower().replace('+', '_')}"
    )
)

agent = ResidualRL(
    models=models,
    models_ref=models_ref,
    memory=memory,
    expert_memory=expert_memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        parser.error(f"ResidualRL checkpoint not found: {args.checkpoint}")
    agent.load(args.checkpoint)
    logger.info(f"Loaded ResidualRL checkpoint: {args.checkpoint}")

trainer = SequentialTrainer(
    env=env,
    agents=agent,
    cfg={"timesteps": args.timesteps, "headless": True},
)

try:
    trainer.eval() if args.eval else trainer.train()
except KeyboardInterrupt:
    logger.info("Stopping ResidualRL training")
finally:
    rtde_c.speedStop()
    time.sleep(0.5)
