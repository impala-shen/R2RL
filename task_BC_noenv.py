import argparse
import os

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gymnasium as gym
import numpy as np
# import new rl agents
from algorithms.BC.BC import BC, BC_CFG
from OfflineTrainer import OfflineTrainer
from envs.snapfit_env import SnapFitRobotEnv  


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")

# --- Space & device config used when env=None ---
NO_ENV_OBS_DIM    = 36    # observation dimension  ← adjust as needed
NO_ENV_STATE_DIM  = 36    # state dimension         ← adjust as needed
NO_ENV_ACTION_DIM = 8     # action dimension        ← adjust as needed
NO_ENV_DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------

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
    
class StochasticActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
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

    def compute(self, inputs, role):
        x = self.net(inputs["observations"])

        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)

        return mean, {"log_std": log_std}
    

# ── helper: build gym spaces manually ────────────────────────────────────────
def make_spaces(obs_dim, state_dim, action_dim):
    """Return (observation_space, state_space, action_space) as gym.spaces."""
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    state_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-np.ones(action_dim, dtype=np.float32)*3.14,
        high= np.ones(action_dim, dtype=np.float32)*3.14,
        dtype=np.float32,
    )
    return observation_space, state_space, action_space
# ─────────────────────────────────────────────────────────────────────────────


# ── build env ────────────────────────────────────────────────────────────────
task_name= "snapfit"
raw_env = SnapFitRobotEnv(rtde_c, rtde_r, home_pose=HOME_POSE, verbose=False, offline_mode = False, device=DEVICE)
env     = wrap_env(raw_env)                         # skrl wrapper
device  = env.device

obs_space    = env.observation_space
state_space  = env.state_space
action_space = env.action_space

# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(43)  # e.g. `set_seed(42)` for fixed seed

# Instantiate a RandomMemory (without replacement) as shared experience replay memory
expert_memory = RandomMemory(memory_size=7500, num_envs=1, device=device, replacement=False)

# # instantiate the agent's models, Behavior clone (BC) requires 1 model to learn the expert behavior
models_BC = {}
models_BC["policy"] = BCmodel(observation_space, state_space, action_space, device)
cfg = BC_CFG()
cfg.gradient_steps = 2
cfg.batch_size = 64
# cfg.demo_file = "./demos/memories/Lift_exp.csv"
cfg.demo_file = "./demos/memories/lift_exp_full.csv"
cfg.reco_demo = False
cfg.exploration_noise = GaussianNoise
cfg.exploration_noise_kwargs = {"mean": 0.0, "std": 0.1, "device": device}
cfg.exploration_scheduler = lambda timestep, timesteps: max(1 - timestep / timesteps, 1e-2)
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = 3e-4
cfg.smooth_regularization_noise = GaussianNoise
cfg.smooth_regularization_noise_kwargs = {"mean": 0.0, "std": 0.0001, "device": device}
cfg.smooth_regularization_clip = 0.005
cfg.learning_starts = 0
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}"

agent = BC(
    models=models_BC,
    memory=expert_memory,
    cfg=cfg,
    observation_space= observation_space,
    state_space= state_space,
    action_space= action_space,
    device=device,
)

# ── Trainer ───────────────────────────────────────────────────────────────────
if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer = OfflineTrainer(agent=agent, cfg={"timesteps": 10000})
trainer.train() if not args.eval else trainer.eval()

# if env is None:
#     # No environment: run offline training loop directly on the agent
#     print("[no-env mode] Starting offline (env-free) training …")
#     timesteps = 5000
#     agent.init()
#     for timestep in range(timesteps):
#         agent.record_transition(observations=None,
#             states=None, actions=None, rewards=None,
#             next_observations=None,
#             next_states=None, terminated=None, truncated=None,
#             infos={}, timestep=timestep, timesteps=timesteps,
#         )

#         agent.pre_interaction(timestep=timestep, timesteps=timesteps)
#         agent.post_interaction(timestep=timestep, timesteps=timesteps)
#         # if timestep % agent.write_interval == 0:
#         agent.write_checkpoint(timestep=timestep, timesteps=timesteps)
#     print("[no-env mode] Training complete.")
