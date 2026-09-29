from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
# from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer
# from skrl.utils import postprocessing

from .BC_cfg import BC_CFG
from algorithms import postprocessing
from .base_noenv import Agent

class BC(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: BC_CFG | dict = {},
    ) -> None:
        """Twin Delayed DDPG (BC).

        https://arxiv.org/abs/1802.09477

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: BC_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=BC_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self._demo_file = self.cfg.demo_file
        self._reco_demo = self.cfg.reco_demo
        self._gradient_steps = self.cfg.gradient_steps
        self._batch_size = self.cfg.batch_size
        self._discount_factor = self.cfg.discount_factor
        self._polyak = self.cfg.polyak


        self._learning_rate_scheduler = self.cfg.learning_rate_scheduler


        self._smooth_regularization_noise = self.cfg.smooth_regularization_noise
        self._smooth_regularization_clip = self.cfg.smooth_regularization_clip

        self._grad_norm_clip = self.cfg.grad_norm_clip
        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()

        # set up noise
        # - exploration noise
        if self.cfg.exploration_noise is not None:
            self._exploration_noise = self.cfg.exploration_noise(**self.cfg.exploration_noise_kwargs)
        else:
            logger.warning("agents:BC: No exploration noise specified, training performance may be degraded")
            self._exploration_noise = None
        # - smooth regularization noise
        if self.cfg.smooth_regularization_noise is not None:
            self._smooth_regularization_noise = self.cfg.smooth_regularization_noise(
                **self.cfg.smooth_regularization_noise_kwargs
            )
        else:
            logger.warning("agents:BC: No smooth regularization noise specified, training variance may be high")
            self._smooth_regularization_noise = None

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizers and learning rate schedulers
        if self.policy is not None:
            # - optimizers
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            # - learning rate schedulers
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    self.policy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )


        # set up preprocessors
        # - observations
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "rewards",
                "next_observations",
                "next_states",
                "terminated",
                "truncated",
            ]
            if not self._reco_demo:
                memory = postprocessing.MemoryFileIterator(self._demo_file)
                for j, data in memory:
                    keys = list(data.keys())
                    N = len(data[keys[0]])
                    for i in range(N-1):
                        self.memory.add_samples(observations=torch.Tensor(np.array(data[keys[1]][i])),
                                                states=torch.Tensor(np.array(data[keys[1]][i])),
                                                    actions=torch.Tensor(np.array(data[keys[0]][i])),
                                                    rewards=torch.zeros(1),
                                                    next_states=torch.Tensor(np.array(data[keys[1]][i])),
                                                    next_observations=torch.Tensor(np.array(data[keys[1]][i])),
                                                    terminated=torch.zeros(1, dtype=torch.float32),
                                                    truncated=torch.zeros(1, dtype=torch.float32))

                print("load memory successfully")
                
                expert_states_r = self.memory.sample(names=["observations"],
                                                            batch_size=N-500)[0][0]
                self.demo_mean0 = torch.mean(expert_states_r, dim=0)
                cov = torch.cov(expert_states_r.T)
                self.demo_cov = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0], device= cov.device))

        # Todo need to be done in the task.
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)
            self.action_scale = (self.clip_actions_max - self.clip_actions_min)/2
            self.action_bias = (self.clip_actions_min + self.clip_actions_max)/2
            # self.clip_actions_min = self.clip_actions_min*3.14
            # self.clip_actions_max = self.clip_actions_max*3.14
        print('self.clip_actions_min', self.clip_actions_min)
        print('self.clip_actions_max', self.clip_actions_max)

        # create temporary variables needed for storage and computation
        self._update_counter = 0

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :para states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        # sample random actions
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
    
        if not self._reco_demo:
            #  For detect state OOD 
            diff = observations - self.demo_mean0
            left = torch.matmul(diff, self.demo_cov)
            dist_sq = (left*diff).sum(dim=1)
            M_dist = torch.sqrt(dist_sq)

            self.track_data("Loss / states BC loss", torch.mean(M_dist).item())

        # add exploration noise
        if self._exploration_noise:
            noises = self._exploration_noise.sample(actions.shape)
            if self.cfg.exploration_scheduler:
                noises.mul_(self.cfg.exploration_scheduler(timestep, timesteps))
            actions = actions * self.action_scale + self.action_bias
            actions.add_(noises)

            self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
            self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
            self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item())

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.memory is not None:
            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

        if self._reco_demo:
            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_observations,
                terminated=terminated,
                truncated=truncated,
            )

            if timestep == timesteps - 1:
                self.memory.save("demos",format="csv")

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if timestep >= self.cfg.learning_starts:
            with ScopedTimer() as timer:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """

        
        # gradient steps
        for gradient_step in range(self._gradient_steps):
            # Sample expert buffer
        # sample a batch from memory
            (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_observations,
                sampled_next_states,
                sampled_terminated,
                truncated
            ) = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]
            # print(sampled_states)

            sampled_states = self._state_preprocessor(sampled_states, train=not gradient_step)
            # sample noises
            noises = self._exploration_noise.sample(sampled_observations.shape)
            sampled_observations.add_(noises)

            inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                    "states": self._state_preprocessor(sampled_states, train=True),
                }
            
            # Compute actor loss
            actions, _ = self.policy.act(inputs, role="policy")
            actions = actions * self.action_scale + self.action_bias
            bc_loss = F.mse_loss(actions, sampled_actions)


            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            self.scaler.scale(bc_loss).backward()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
            self.scaler.step(self.policy_optimizer)

            self.scaler.update()

            # update learning rate
            if self.policy_scheduler:
                self.policy_scheduler.step()

            self.track_data("Loss / Policy loss", bc_loss.item())
