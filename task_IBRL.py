import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# import the skrl components to build the RL system
from skrl import logger
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
# from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# import new rl agents
from algorithms.IBRL.ibrl import IBRL, IBRL_CFG
from task_settings import TASKS, add_task_arguments, resolve_demo, validate_demo, create_robot_env
from models.gaussian import GaussianMixin

# import ur interface
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq.robotiq_gripper_control import RobotiqGripper

# Select defaults here, or override them on the command line.
DEFAULT_TASK = "assembly"
BC_CHECKPOINT_PATH = None

# parse arguments
parser = argparse.ArgumentParser()
add_task_arguments(parser, default_task=DEFAULT_TASK)
parser.add_argument("--bc-checkpoint", default=BC_CHECKPOINT_PATH, help="Frozen imitation-policy checkpoint matching the task and architecture")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
args, _ = parser.parse_known_args()

task = TASKS[args.task]
if args.max_steps is not None and args.max_steps < 1:
    parser.error("--max-steps must be positive")
if not args.bc_checkpoint or not os.path.isfile(args.bc_checkpoint):
    parser.error("Set BC_CHECKPOINT_PATH or supply an existing --bc-checkpoint for the selected task")
if args.checkpoint and not os.path.isfile(args.checkpoint):
    parser.error(f"IBRL checkpoint not found: {args.checkpoint}")
demo_file = resolve_demo(task, args.demo_file, ibrl=True)
try:
    demo_rows = validate_demo(demo_file, task)
except (OSError, ValueError) as error:
    parser.error(str(error))

set_seed(42)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================
# define TD3 models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Tanh()
        )

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
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1)), {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
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

    
# Define the BC NN architecture (stochastic and deterministic models) for the agents using mixins.
# - DeterministicActor: takes as input the environment's observation/state and returns an action
class BCmodel(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            # nn.Tanh(),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# Shared task selection keeps the environment, home pose, and data together.
ROBOT_IP = args.robot_ip
rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)
rtde_c.zeroFtSensor()
gripper_ = RobotiqGripper(rtde_c) if task.requires_gripper else None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_STEPS = args.max_steps if args.max_steps is not None else task.max_steps
task_name = args.task
raw_env = create_robot_env(
    task, rtde_c, rtde_r, gripper=gripper_, device=DEVICE,
    home_pose=args.home_pose, max_steps=MAX_STEPS,
)
logger.info(f"Task: {task_name}, home pose: {raw_env.home}")
demo_length = MAX_STEPS
env     = wrap_env(raw_env)                         # skrl wrapper
device  = env.device

obs_space    = env.observation_space
state_space  = env.state_space
action_space = env.action_space


# instantiate a replay memory
memory = RandomMemory(memory_size=10000, num_envs=1, device=device)
expert_memory = RandomMemory(memory_size=demo_rows, num_envs=1, device=device)

# ============================================================================
# # Initiate an IL agent in order to load a trained BC agent
# ============================================================================
# Behavior clone (BC) requires 1 model to learn the expert behavior, directly load
models_BC = {}
models_BC["policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device)
# Load the full saved state
saved_state = torch.load(args.bc_checkpoint, map_location=device)
# Extract just the policy weights
policy_state = saved_state["policy"]
models_BC["policy"].load_state_dict(policy_state)
models_BC["policy"].eval()



# Create Emseabling Q networks
deep_ensemble = True
ensemble_size = 5
# Create ensemble of critics (each with unique parameters)
critics = []
for i in range(ensemble_size):
    # Create new critic instance for each position
    critic = Critic(env.observation_space, env.state_space, env.action_space, device)

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    critic.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    critics.append(critic)

# instantiate the agent's models (function approximators).
# TD3 requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#models
models = {}
# models["policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device)
models["policy"] = StochasticActor(env.observation_space, env.state_space, env.action_space, device)
if deep_ensemble:
    for i in range(ensemble_size):
        models[f"critic_{i+1}"] = critics[i]
else:
    models["critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["value"] = Value(env.observation_space, env.state_space, device)
models["target_policy"] = StochasticActor(env.observation_space, env.state_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#configuration-and-hyperparameters
cfg = IBRL_CFG()
cfg.exploration_noise = GaussianNoise
cfg.exploration_noise_kwargs = {"mean": 0.0, "std": 0.1, "device": device}
cfg.exploration_scheduler = lambda timestep, timesteps: max(1 - timestep / timesteps, 1e-2)
cfg.smooth_regularization_noise = GaussianNoise
cfg.smooth_regularization_noise_kwargs = {"mean": 0.0, "std": 0.2, "device": device}
cfg.smooth_regularization_clip = 0.5
cfg.gradient_steps = 3
cfg.batch_size = 64
cfg.suc_score = 1000  # 1000 for snapfit, 1 for openbox
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = 1e-4
cfg.random_timesteps = demo_length*1
cfg.episode_steps = demo_length
cfg.learning_starts = 0
cfg.expectile = 0.7  #0.71
cfg.observation_preprocessor_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.num_envs = env.num_envs
cfg.select_act = True   # Flag to enable IBRL style data incorparation, set False to use RLPD style
cfg.deterministic_policy = False
cfg.learn_entropy = False
cfg.initial_entropy_value = 0.1
cfg.demo_file = str(demo_file)
cfg.Deep_Ensemble = deep_ensemble
cfg.Ensemble_size = ensemble_size
cfg.select_method = "on_state"  # select from "iql_v","off_state","adv" or else (IQL-style expectile regression), "bellman" (standard Bellman update), "cql" (CQL-style conservative update)
cfg.q_method = "bellman"    # Select from iql, cql, or standard bellman.
cfg.policy_method = "maxQ"  # Select from DDPG+BC, AWR style or simpliy maxQ.
cfg.OPEX = False
cfg.opex_beta = 0.001
cfg.TTT = False
cfg.uncertainty_epsilon = 0.  ## Hyperparameter for controlling the uncertainty penalty (higher values lead to more conservative behavior)
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = 100 if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}/exp1"

agent = IBRL(
    models=models,
    models_ref=models_BC,
    memory=memory,
    expert_memory=expert_memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# ── optional: warm-start from your offline checkpoint ─────────────────────────
if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        exit(1)
    agent.load(args.checkpoint)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

# ── trainer ───────────────────────────────────────────────────────────────────
trainer_cfg = {
    "timesteps":       demo_length*100,     # total env steps — adjust to your budget
    "headless":        True,
}

trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

try:
    trainer.train() if not args.eval else trainer.eval()
except KeyboardInterrupt:
    logger.info("KeyboardInterrupt caught — stopping trainer...")
finally:
    logger.info("Stopping robot...")
    rtde_c.speedStop()
    time.sleep(0.5)
    logger.info("Robot stopped and disconnected.")