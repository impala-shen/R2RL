"""Train OSRL or evaluate a trained policy checkpoint on the robot.

Evaluation uses the stochastic policy mean by default::

    python task_OSRL.py --eval --checkpoint path/to/agent_5000.pt

Pass ``--stochastic-evaluation`` only when sampled evaluation is intentional.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
# from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# import new rl agents
from algorithms.OneShot.osrl import OSRL, OSRL_CFG
from envs.snapfit_env import SnapFitRobotEnv  
from envs.open_box import OpenBoxRobotEnv
from envs.assembly import AssemblyRobotEnv
from envs.peginhole import PegInHoleRobotEnv
from models.gaussian import GaussianMixin

# import ur interface
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq.robotiq_gripper_control import RobotiqGripper

# Set this directly to a trained OSRL ``agent_*.pt`` checkpoint, or override
# it with ``--checkpoint`` on the command line.
OSRL_CHECKPOINT_PATH = "./runs/torch/snapfit/osrl/OSRL_new/checkpoints/best_agent.pt"
EVALUATION_EPISODES = 30

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default=OSRL_CHECKPOINT_PATH,
    help="Trained OSRL agent checkpoint (overrides OSRL_CHECKPOINT_PATH)",
)
parser.add_argument(
    "--eval",
    action="store_true",
    help="Evaluate the checkpointed policy without replay selection or training",
)
parser.add_argument(
    "--eval-episodes",
    type=int,
    default=EVALUATION_EPISODES,
    help="Number of episodes to run with --eval",
)
parser.add_argument(
    "--stochastic-evaluation",
    action="store_true",
    help="Sample the stochastic policy during evaluation instead of using its mean",
)
args, _ = parser.parse_known_args()

if args.checkpoint and not os.path.isfile(args.checkpoint):
    parser.error(f"OSRL checkpoint not found: {args.checkpoint}")
if args.eval and not args.checkpoint:
    parser.error("Set OSRL_CHECKPOINT_PATH or pass --checkpoint when using --eval")
if args.eval_episodes < 1:
    parser.error("--eval-episodes must be at least 1")

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
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.Tanh(),   ## For saturate the output in between [-1,1]
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


# ── helper ──────────────────────────────────────────────────────────
def Get_initTCPpos():
    Init_TCP = rtde_r.getActualTCPPose()
    return Init_TCP

# ── robot connection ──────────────────────────────────────────────────────────
ROBOT_IP  = "192.168.1.11"
rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)
rtde_c.zeroFtSensor()

gripper_ = RobotiqGripper(rtde_c)

init_TCP = np.array(Get_initTCPpos())
# HOME_POSE = init_TCP     #  Safe inital home pose # -0.002
# HOME_POSE = np.array([-0.38633344, -0.22858439,  0.10802976, -3.06064544, -0.6274192,  0.03773944])  ## PeginHole init pos
# HOME_POSE = np.array([-0.39687852,  0.01385717,  0.14660621,  0.99239598,  1.41831132, -0.89998169])  ## assemble init pos 2

HOME_POSE = np.array([-0.35837264, -0.07759438, 0.23013212, -3.04344862, -0.60611295, -0.16027859])

# HOME_POSE = np.array([-0.4982482,  -0.11075026,  0.08202713, -3.04956999, -0.68303232, -0.11320668])  ##open new box
# HOME_POSE = np.array([-0.48796237, -0.17673211,  0.01922106,  2.96961517,  0.5870512,   0.05120328])  ##open box v0

# HOME_POSE = np.array([-0.42646054,  0.10034163,  0.18232037,  3.05363488,  0.61486366,  0.00385519])  ##snapfit2 disasemble
# HOME_POSE = np.array([-4.52200962e-01,  1.59801267e-01,  2.25089998e-01,  2.99534608e+00,  6.31839988e-01, -1.64644767e-03])  ##snapfit2 disasemble
# HOME_POSE = np.array([-0.44172542,  0.1301668,   0.22782574,  3.01452808,  0.6134789,   0.05645406])  ## disasemble snapfit1
print("Safe initial TCP:", HOME_POSE)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 100

# # ── build env ────────────────────────────────────────────────────────────────
task_name = "snapfit"
raw_env = SnapFitRobotEnv(rtde_c, rtde_r, home_pose=HOME_POSE, max_steps=100, verbose=False, offline_mode = False, device=DEVICE)

# task_name = "openbox_v0"
# raw_env = OpenBoxRobotEnv(rtde_c, rtde_r, home_pose=HOME_POSE, max_steps=MAX_STEPS, verbose=False, offline_mode = False, device=DEVICE)

# task_name = "peginhole"
# raw_env = PegInHoleRobotEnv(rtde_c, rtde_r, gripper=gripper_, home_pose=HOME_POSE, max_steps=100, verbose=False, offline_mode = False, device=DEVICE)

# task_name = "assembly"
# raw_env = AssemblyRobotEnv(rtde_c, rtde_r, gripper=gripper_, home_pose=HOME_POSE, max_steps=100, verbose=False, offline_mode = False, device=DEVICE)


demo_length = 100
env     = wrap_env(raw_env)                         # skrl wrapper
device  = env.device

obs_space    = env.observation_space
state_space  = env.state_space
action_space = env.action_space


# Evaluation is policy-only, so it does not allocate or load replay memories.
memory = None if args.eval else RandomMemory(memory_size=MAX_STEPS*100, num_envs=1, device=device)
expert_memory = (
    None
    if args.eval
    else RandomMemory(memory_size=demo_length*1, num_envs=1, device=device)
)


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
cfg = OSRL_CFG()
cfg.exploration_noise = GaussianNoise
cfg.exploration_noise_kwargs = {"mean": 0.0, "std": 0.1, "device": device}
cfg.exploration_scheduler = lambda timestep, timesteps: max(1 - timestep / timesteps, 1e-2)
cfg.smooth_regularization_noise = GaussianNoise
cfg.smooth_regularization_noise_kwargs = {"mean": 0.0, "std": 0.2, "device": device}
cfg.smooth_regularization_clip = 0.5
cfg.gradient_steps = 1
cfg.batch_size = 64
cfg.suc_score = 1000  # 1000 for snapfit, 1 for openbox
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = 1e-4
cfg.random_timesteps = demo_length*1
cfg.episode_steps = demo_length
cfg.learning_starts = 0
cfg.expectile = 0.70  #0.71
cfg.observation_preprocessor_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.num_envs = env.num_envs
cfg.select_act = True   # Flag to enable IBRL style data incorparation, set False to use RLPD style
cfg.deterministic_policy = False
cfg.learn_entropy = False
cfg.initial_entropy_value = 0.1
# cfg.demo_file = "./demos/Peginhole/assembly.csv"
# cfg.demo_file = "./demos/Peginhole/10demos.csv"

# cfg.demo_file = "./demos/snapfit0/10demos.csv"
cfg.demo_file = "./demos/snapfit0/snapfit_fixscale.csv"
# cfg.demo_file = "./demos/snapfit0/one_demo_downsample_dn1.csv"    ## Use only one demonstration for training
cfg.Deep_Ensemble = deep_ensemble
cfg.Ensemble_size = ensemble_size
cfg.select_method = "on_state"  # select from "iql_v","off_state","adv" or else (IQL-style expectile regression), "bellman" (standard Bellman update), "cql" (CQL-style conservative update)
cfg.q_method = "iql"    # Select from iql, cql, or standard bellman.
cfg.policy_method = "DDPG+BC"  # Select from DDPG+BC, AWR style or simpliy maxQ.
cfg.OPEX = False
cfg.opex_beta = 0.001
cfg.TTT = True
cfg.uncertainty_epsilon = 0.  ## Hyperparameter for controlling the uncertainty penalty (higher values lead to more conservative behavior)
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = 100 if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}/osrl"

agent = OSRL(
    models=models,
    models_ref=models,
    memory=memory,
    expert_memory=expert_memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# ── load a trained checkpoint for evaluation, or optionally warm-start training ─
if args.checkpoint:
    agent.load(args.checkpoint)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

# ── trainer ───────────────────────────────────────────────────────────────────
trainer_cfg = {
    "timesteps":       MAX_STEPS * (args.eval_episodes if args.eval else 50),
    "headless":        True,
    "stochastic_evaluation": args.stochastic_evaluation,
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
