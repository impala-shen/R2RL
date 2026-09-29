from __future__ import annotations

from typing import Any
import itertools

import gymnasium
from packaging import version
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skrl import config, logger
# from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from algorithms import postprocessing
from .residual_rl_cfg import ResidualRL_CFG
from .base_c import Agent

class ResidualRL(Agent):
    """Residual RL with independently selectable actor and critic updates.

    The environment executes ``clip(pi_base(s) + pi_residual(s))``. Following
    Algorithm 1 of Johannink et al. (2018), replay stores the residual action,
    because that is the action optimized by the RL policy; the base policy is a
    fixed part of the induced environment dynamics.

    Paper: https://arxiv.org/abs/1812.03201
    """

    def __init__(
        self,
        *,
        models: dict[str, Model],
        models_ref: dict[str, Model],
        memory: Memory | None = None,
        expert_memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: ResidualRL_CFG | dict = {},
    ) -> None:
        self.cfg: ResidualRL_CFG
        super().__init__(
            models=models,
            models_ref=models_ref,
            memory=memory,
            expert_memory=expert_memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=ResidualRL_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )
        self.cfg.validate()
        self._q_method = self.cfg.q_method
        self._policy_method = self.cfg.policy_method

        self.policy = self.models.get("policy")
        self.target_policy = self.models.get("target_policy")
        self.critic_1 = self.models.get("critic_1")
        self.critic_2 = self.models.get("critic_2")
        self.target_critic_1 = self.models.get("target_critic_1")
        self.target_critic_2 = self.models.get("target_critic_2")
        self.value = self.models.get("value")

        required = {
            "policy": self.policy,
            "critic_1": self.critic_1,
            "critic_2": self.critic_2,
        }
        if self._q_method == "bellman":
            required["target_critic_1"] = self.target_critic_1
            required["target_critic_2"] = self.target_critic_2
        if self.cfg.deterministic_policy and self._q_method == "bellman":
            required["target_policy"] = self.target_policy
        if self._q_method == "iql":
            required["value"] = self.value
        missing = [name for name, model in required.items() if model is None]
        if missing:
            raise KeyError(f"Missing ResidualRL models: {', '.join(missing)}")

        for name, model in required.items():
            self.checkpoint_modules[name] = model

        self.base_policy_type = self.cfg.base_policy_type
        self.base_policy = None
        if self.base_policy_type == "checkpoint":
            self.base_policy = models_ref.get("policy")
            if self.base_policy is None:
                raise KeyError("models_ref['policy'] is required for checkpoint base-policy mode")
            self.base_policy.to(self.device)
            self.base_policy.eval()
            self.base_policy.freeze_parameters(True)
            self.checkpoint_modules["base_policy"] = self.base_policy
        elif self.expert_memory is None:
            raise ValueError("expert_memory is required for one-demo base-policy mode")

        self._demo_file = self.cfg.demo_file
        self._num_envs = self.cfg.num_envs
        self.episode_done = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self.epi_timestep = 0
        self._demo_actions: torch.Tensor | None = None

        if config.torch.is_distributed:
            logger.info("Broadcasting ResidualRL models' parameters")
            for model in required.values():
                model.broadcast_parameters()

        self._exploration_noise = (
            self.cfg.exploration_noise(**self.cfg.exploration_noise_kwargs)
            if self.cfg.exploration_noise is not None
            else None
        )
        self._smooth_regularization_noise = (
            self.cfg.smooth_regularization_noise(**self.cfg.smooth_regularization_noise_kwargs)
            if self.cfg.smooth_regularization_noise is not None
            else None
        )

        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
        self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
        self.policy_scheduler = self._make_scheduler(0, self.policy_optimizer)

        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
            lr=self.cfg.learning_rate[1],
        )
        self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
        self.critic_scheduler = self._make_scheduler(1, self.critic_optimizer)

        self.value_optimizer = None
        if self._q_method == "iql":
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.cfg.learning_rate[1])
            self.checkpoint_modules["value_optimizer"] = self.value_optimizer

        self._entropy_coefficient: float | torch.Tensor = self.cfg.initial_entropy_value
        self._target_entropy = self.cfg.target_entropy
        self.entropy_coefficient_module = None
        self.log_entropy_coefficient = None
        self.entropy_optimizer = None
        if not self.cfg.deterministic_policy and self.cfg.learn_entropy:
            if self._target_entropy is None:
                self._target_entropy = -float(np.prod(self.action_space.shape))
            self.entropy_coefficient_module = nn.ParameterDict(
                {
                    "log_entropy_coefficient": nn.Parameter(
                        torch.log(
                            torch.ones(1, device=self.device) * self.cfg.initial_entropy_value
                        )
                    )
                }
            )
            self.log_entropy_coefficient = self.entropy_coefficient_module[
                "log_entropy_coefficient"
            ]
            self.entropy_optimizer = torch.optim.Adam(
                [self.log_entropy_coefficient], lr=self.cfg.entropy_learning_rate
            )
            self.checkpoint_modules["entropy_coefficient"] = self.entropy_coefficient_module
            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        if self.target_policy is not None:
            self.target_policy.freeze_parameters(True)
        if self.target_critic_1 is not None:
            self.target_critic_1.freeze_parameters(True)
        if self.target_critic_2 is not None:
            self.target_critic_2.freeze_parameters(True)
        if self.target_policy is not None:
            self.target_policy.update_parameters(self.policy, polyak=1)
        if self.target_critic_1 is not None:
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
        if self.target_critic_2 is not None:
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor

        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        self._last_residual_actions: torch.Tensor | None = None

    def _make_scheduler(self, index: int, optimizer: torch.optim.Optimizer):
        scheduler_cls = self.cfg.learning_rate_scheduler[index]
        if scheduler_cls is None:
            return None
        return scheduler_cls(optimizer, **self.cfg.learning_rate_scheduler_kwargs[index])

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)
        if self.base_policy is not None:
            self.base_policy.eval()

        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self._tensors_names = [
                "observations", "states", "actions", "rewards",
                "next_observations", "next_states", "terminated", "truncated",
            ]

        # Create and load the open-loop base trajectory only in one-demo mode.
        if self.base_policy_type == "one_demo":
            self.expert_memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.expert_memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.expert_memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.expert_memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self.expert_memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.expert_memory.create_tensor(name="next_actions", size=self.action_space, dtype=torch.float32)
            self.expert_memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.expert_memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.expert_memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names_exp = [
                "observations",
                "states",
                "actions",
                "rewards",
                "next_observations",
                "next_states",
                "next_actions",
                "terminated",
                "truncated",
            ]
            expert_memory = postprocessing.MemoryFileIterator(self._demo_file)
            
            demo_action_count = 0
            for j, data in expert_memory:
                keys = list(data.keys())
                N = len(data[keys[0]])
                demo_action_count += N
                for i in range(0, N):
                    self.expert_memory.add_samples(
                                                   observations=torch.Tensor(np.array(data[keys[1]][i])),
                                                   states=torch.Tensor(np.array(data[keys[1]][i])),
                                                   actions=torch.Tensor(np.array(data[keys[0]][i])),
                                                   rewards=torch.Tensor(np.array(data[keys[3]][i])),
                                                   next_states=torch.Tensor(np.array(data[keys[2]][i])),
                                                   next_observations=torch.Tensor(np.array(data[keys[2]][i])),
                                                   next_actions=torch.Tensor(np.array(data[keys[0]][i])),
                                                   terminated=torch.tensor(np.array(data[keys[4]][i]), dtype=torch.float32),
                                                   truncated=torch.tensor(np.array(data[keys[5]][i]), dtype=torch.float32))
         
            # self.expert_memory.save("demos", format="csv")
                    # print(data[keys[6]][i])
            print("load expert memory successfully")
            # expert_returns = self.expert_memory.sample(names=["returns"],
            #                                             batch_size=N)[0][0]
            # self.demo_mean = torch.mean(expert_returns, dim=0)

            self._demo_actions = self.expert_memory.sample_all(
                names=["actions"], mini_batches=1, sequence_length=1
            )[0][0][:demo_action_count]

        if self.action_space is None or not isinstance(self.action_space, gymnasium.spaces.Box):
            raise TypeError("ResidualRL requires a bounded gymnasium.spaces.Box action space")
        self.action_min = torch.as_tensor(self.action_space.low, device=self.device, dtype=torch.float32)
        self.action_max = torch.as_tensor(self.action_space.high, device=self.device, dtype=torch.float32)
        if not torch.isfinite(self.action_min).all() or not torch.isfinite(self.action_max).all():
            raise ValueError("ResidualRL requires finite action bounds")
        self.action_scale = (self.action_max - self.action_min) / 2
        self.action_bias = (self.action_max + self.action_min) / 2
        self.residual_limit = self.action_scale * self.cfg.residual_action_scale
        self._update_counter = 0

    def _policy_to_residual(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return normalized_action.clamp(-1, 1) * self.residual_limit

    def _base_action(self, observations: torch.Tensor, states: torch.Tensor | None) -> torch.Tensor:
        if self.cfg.preprocess_base_observations:
            base_inputs = {
                "observations": self._observation_preprocessor(observations),
                "states": self._state_preprocessor(states),
            }
        else:
            base_inputs = {"observations": observations, "states": states}
        with torch.no_grad():
            base_actions, _ = self.base_policy.act(base_inputs, role="policy")
        if self.cfg.base_policy_outputs_normalized:
            base_actions = base_actions.clamp(-1, 1) * self.action_scale + self.action_bias
        return base_actions

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        if self.base_policy_type == "one_demo":
            if self._demo_actions is None or not len(self._demo_actions):
                raise RuntimeError("No demonstration actions were loaded")
            if self.episode_done.any():
                self.epi_timestep = 0
            demo_index = min(self.epi_timestep, self._demo_actions.shape[0] - 1)
            base_actions = self._demo_actions[demo_index].repeat(observations.shape[0], 1)
            self.epi_timestep += 1
            self.episode_done.zero_()
        else:
            base_actions = self._base_action(observations, states)

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            if self.training and timestep < self.cfg.random_timesteps:
                residual_actions = torch.zeros_like(base_actions)
                outputs = {}
            else:
                normalized_residual, outputs = self.policy.act(inputs, role="policy")
                if not self.cfg.deterministic_policy and not self.training:
                    normalized_residual = outputs["mean_actions"]
                residual_actions = self._policy_to_residual(normalized_residual)

        if (
            self.cfg.deterministic_policy
            and self.training
            and self._exploration_noise is not None
            and timestep >= self.cfg.random_timesteps
        ):
            noise = self._exploration_noise.sample(residual_actions.shape)
            if self.cfg.exploration_scheduler is not None:
                noise.mul_(self.cfg.exploration_scheduler(timestep, timesteps))
            residual_actions = residual_actions.add(noise).clamp(-self.residual_limit, self.residual_limit)
            self.track_data("Exploration / Residual noise (mean)", noise.mean().item())

        combined_actions = (base_actions + residual_actions).clamp(self.action_min, self.action_max)
        self._last_residual_actions = residual_actions.detach().clone()
        outputs = dict(outputs)
        if not self.cfg.deterministic_policy and "mean_actions" in outputs:
            mean_residual = self._policy_to_residual(outputs["mean_actions"])
            outputs["mean_actions"] = (base_actions + mean_residual).clamp(
                self.action_min, self.action_max
            )
        outputs.update(
            base_actions=base_actions.detach(),
            residual_actions=residual_actions.detach(),
            unclipped_actions=(base_actions + residual_actions).detach(),
        )

        base_magnitude = base_actions.norm(dim=-1)
        residual_magnitude = residual_actions.norm(dim=-1)

        intervention_ratio = (
            residual_magnitude
            / (base_magnitude + residual_magnitude + 1e-6)
        )

        self.track_data(
            "Action / residual intervention ratio",
            intervention_ratio.mean().item(),
)
        self.track_data("Action / Base magnitude", base_magnitude.mean().item())
        self.track_data("Action / Residual magnitude", residual_magnitude.mean().item())
        self.track_data(
            "Action / Combined clip rate",
            ((base_actions + residual_actions) != combined_actions).float().mean().item(),
        )
        return combined_actions, outputs

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
        super().record_transition(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=timestep, timesteps=timesteps,
        )
        self.episode_done = (terminated | truncated).reshape(-1).bool()
        if not self.training or self.memory is None:
            return
        if self._last_residual_actions is None:
            raise RuntimeError("record_transition called before ResidualRL.act")
        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
        self.memory.add_samples(
            observations=observations,
            states=states,
            actions=self._last_residual_actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
        )
        if timestep == timesteps - 1:
            self.memory.save("demos/residual_rl", format="csv")

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.training and timestep >= self.cfg.learning_starts and self.memory is not None:
            with ScopedTimer() as timer:
                self.enable_models_training_mode(True)
                if self.base_policy is not None:
                    self.base_policy.eval()
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        # Keep the original TD3 update as the exact default/preservation path.
        if (
            self.cfg.deterministic_policy
            and self._q_method == "bellman"
            and self._policy_method == "maxQ"
        ):
            self._update_td3(timestep=timestep, timesteps=timesteps)
        else:
            self._update_composed(timestep=timestep, timesteps=timesteps)

    def _update_td3(self, *, timestep: int, timesteps: int) -> None:
        for gradient_step in range(self.cfg.gradient_steps):
            (
                sampled_observations,
                sampled_states,
                sampled_residuals,
                sampled_rewards,
                sampled_next_observations,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]

            inputs = {
                "observations": self._observation_preprocessor(sampled_observations, train=not gradient_step),
                "states": self._state_preprocessor(sampled_states, train=not gradient_step),
            }
            next_inputs = {
                "observations": self._observation_preprocessor(sampled_next_observations, train=not gradient_step),
                "states": self._state_preprocessor(sampled_next_states, train=not gradient_step),
            }

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                with torch.no_grad():
                    next_normalized_residuals, _ = self.target_policy.act(next_inputs, role="target_policy")
                    next_residuals = self._policy_to_residual(next_normalized_residuals)
                    if self._smooth_regularization_noise is not None:
                        noise = self._smooth_regularization_noise.sample(next_residuals.shape).clamp(
                            -self.cfg.smooth_regularization_clip,
                            self.cfg.smooth_regularization_clip,
                        )
                        next_residuals = (next_residuals + noise).clamp(
                            -self.residual_limit, self.residual_limit
                        )
                    target_q1, _ = self.target_critic_1.act(
                        {**next_inputs, "taken_actions": next_residuals}, role="target_critic_1"
                    )
                    target_q2, _ = self.target_critic_2.act(
                        {**next_inputs, "taken_actions": next_residuals}, role="target_critic_2"
                    )
                    not_done = (sampled_terminated | sampled_truncated).logical_not()
                    target_values = sampled_rewards + self.cfg.discount_factor * not_done * torch.min(
                        target_q1, target_q2
                    )

                critic_1_values, _ = self.critic_1.act(
                    {**inputs, "taken_actions": sampled_residuals}, role="critic_1"
                )
                critic_2_values, _ = self.critic_2.act(
                    {**inputs, "taken_actions": sampled_residuals}, role="critic_2"
                )
                critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(
                    critic_2_values, target_values
                )

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self.cfg.grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)

            self._update_counter += 1
            if self._update_counter % self.cfg.policy_delay == 0:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    normalized_residuals, _ = self.policy.act(inputs, role="policy")
                    policy_residuals = self._policy_to_residual(normalized_residuals)
                    policy_q, _ = self.critic_1.act(
                        {**inputs, "taken_actions": policy_residuals}, role="critic_1"
                    )
                    policy_loss = -policy_q.mean()

                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                self.scaler.step(self.policy_optimizer)

                self.target_policy.update_parameters(self.policy, polyak=self.cfg.polyak)
                self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)
                self.track_data("Loss / Policy loss", policy_loss.item())

            self.scaler.update()
            if self.policy_scheduler is not None:
                self.policy_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()

            self.track_data("Loss / Critic loss", critic_loss.item())
            self.track_data("Q-network / Q1 (mean)", critic_1_values.mean().item())
            self.track_data("Q-network / Q2 (mean)", critic_2_values.mean().item())
            self.track_data("Target / Target (mean)", target_values.mean().item())

    def _sample_training_batch(self, gradient_step: int):
        batch = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]
        (
            sampled_observations,
            sampled_states,
            sampled_residuals,
            sampled_rewards,
            sampled_next_observations,
            sampled_next_states,
            sampled_terminated,
            sampled_truncated,
        ) = batch
        inputs = {
            "observations": self._observation_preprocessor(
                sampled_observations, train=not gradient_step
            ),
            "states": self._state_preprocessor(sampled_states, train=not gradient_step),
        }
        next_inputs = {
            "observations": self._observation_preprocessor(
                sampled_next_observations, train=not gradient_step
            ),
            "states": self._state_preprocessor(sampled_next_states, train=not gradient_step),
        }
        not_done = (sampled_terminated | sampled_truncated).logical_not()
        return inputs, next_inputs, sampled_residuals, sampled_rewards, not_done

    def _step_critic(self, critic_loss: torch.Tensor) -> None:
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        if config.torch.is_distributed:
            self.critic_1.reduce_parameters()
            self.critic_2.reduce_parameters()
        if self.cfg.grad_norm_clip > 0:
            self.scaler.unscale_(self.critic_optimizer)
            nn.utils.clip_grad_norm_(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                self.cfg.grad_norm_clip,
            )
        self.scaler.step(self.critic_optimizer)

    def _step_policy(self, policy_loss: torch.Tensor) -> None:
        self.policy_optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward()
        if config.torch.is_distributed:
            self.policy.reduce_parameters()
        if self.cfg.grad_norm_clip > 0:
            self.scaler.unscale_(self.policy_optimizer)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
        self.scaler.step(self.policy_optimizer)

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * diff.square()).mean()

    def _residual_to_policy(self, residuals: torch.Tensor) -> torch.Tensor:
        safe_limit = torch.where(
            self.residual_limit.abs() > 1e-8,
            self.residual_limit,
            torch.ones_like(self.residual_limit),
        )
        normalized = residuals / safe_limit
        normalized = torch.where(
            self.residual_limit.abs() > 1e-8, normalized, torch.zeros_like(normalized)
        )
        return normalized.clamp(-0.999999, 0.999999)

    def _update_composed(self, *, timestep: int, timesteps: int) -> None:
        """OSRL-style composition of critic target and actor objective.

        ``q_method`` controls only the critic/value target. ``policy_method``
        controls only the actor loss, and ``deterministic_policy`` controls
        whether the actor emits a point action or a squashed Gaussian sample.
        """
        for gradient_step in range(self.cfg.gradient_steps):
            inputs, next_inputs, residuals, rewards, not_done = self._sample_training_batch(
                gradient_step
            )

            value_loss = None
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                if self._q_method == "iql":
                    # V(s) uses only replay actions; the actor objective remains
                    # independently selectable below.
                    with torch.no_grad():
                        q1_data, _ = self.critic_1.act(
                            {**inputs, "taken_actions": residuals}, role="critic_1"
                        )
                        q2_data, _ = self.critic_2.act(
                            {**inputs, "taken_actions": residuals}, role="critic_2"
                        )
                        q_data = torch.min(q1_data, q2_data)
                    values, _ = self.value.act(inputs, role="value")
                    value_loss = self._expectile_loss(q_data - values, self.cfg.expectile)

            if value_loss is not None:
                self.value_optimizer.zero_grad()
                self.scaler.scale(value_loss).backward()
                if config.torch.is_distributed:
                    self.value.reduce_parameters()
                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.value_optimizer)
                    nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_norm_clip)
                self.scaler.step(self.value_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                with torch.no_grad():
                    if self._q_method == "iql":
                        next_values, _ = self.value.act(next_inputs, role="value")
                        target_values = rewards + self.cfg.discount_factor * not_done * next_values
                    elif self.cfg.deterministic_policy:
                        next_normalized, _ = self.target_policy.act(
                            next_inputs, role="target_policy"
                        )
                        next_residuals = self._policy_to_residual(next_normalized)
                        if self._smooth_regularization_noise is not None:
                            noise = self._smooth_regularization_noise.sample(
                                next_residuals.shape
                            ).clamp(
                                -self.cfg.smooth_regularization_clip,
                                self.cfg.smooth_regularization_clip,
                            )
                            next_residuals = (next_residuals + noise).clamp(
                                -self.residual_limit, self.residual_limit
                            )
                        target_q1, _ = self.target_critic_1.act(
                            {**next_inputs, "taken_actions": next_residuals},
                            role="target_critic_1",
                        )
                        target_q2, _ = self.target_critic_2.act(
                            {**next_inputs, "taken_actions": next_residuals},
                            role="target_critic_2",
                        )
                        target_values = rewards + self.cfg.discount_factor * not_done * torch.min(
                            target_q1, target_q2
                        )
                    else:
                        next_normalized, next_outputs = self.policy.act(
                            next_inputs, role="policy"
                        )
                        next_residuals = self._policy_to_residual(next_normalized)
                        target_q1, _ = self.target_critic_1.act(
                            {**next_inputs, "taken_actions": next_residuals},
                            role="target_critic_1",
                        )
                        target_q2, _ = self.target_critic_2.act(
                            {**next_inputs, "taken_actions": next_residuals},
                            role="target_critic_2",
                        )
                        target_q = torch.min(target_q1, target_q2)
                        if self.cfg.learn_entropy:
                            target_q = target_q - self._entropy_coefficient * next_outputs["log_prob"]
                        target_values = rewards + self.cfg.discount_factor * not_done * target_q

                critic_1_values, _ = self.critic_1.act(
                    {**inputs, "taken_actions": residuals}, role="critic_1"
                )
                critic_2_values, _ = self.critic_2.act(
                    {**inputs, "taken_actions": residuals}, role="critic_2"
                )
                critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(
                    critic_2_values, target_values
                )

            self._step_critic(critic_loss)

            self._update_counter += 1
            if self._update_counter % self.cfg.policy_delay == 0:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    normalized_actions, policy_outputs = self.policy.act(inputs, role="policy")
                    policy_residuals = self._policy_to_residual(normalized_actions)
                    policy_q, _ = self.critic_1.act(
                        {**inputs, "taken_actions": policy_residuals}, role="critic_1"
                    )
                    q_loss = -policy_q.mean()

                    bc_loss = None
                    bc_lambda = None
                    if self._policy_method == "DDPG+BC":
                        if self.cfg.deterministic_policy:
                            bc_loss = F.mse_loss(policy_residuals, residuals)
                        else:
                            # Replay stores physical residuals while the Gaussian
                            # density is defined in normalized tanh-action space.
                            replay_normalized = self._residual_to_policy(residuals)
                            bc_log_prob = self.policy.evaluate(inputs, replay_normalized)
                            bc_loss = -bc_log_prob.mean()
                        bc_lambda = self.cfg.ddpg_bc_alpha / policy_q.abs().mean().detach().clamp_min(
                            1e-6
                        )
                        policy_loss = (
                            bc_lambda * q_loss + self.cfg.bc_loss_scale * bc_loss
                        )
                    else:
                        policy_loss = q_loss

                    log_prob = None
                    if not self.cfg.deterministic_policy:
                        log_prob = policy_outputs["log_prob"]
                        if self.cfg.learn_entropy:
                            policy_loss = policy_loss + self._entropy_coefficient * log_prob.mean()

                self._step_policy(policy_loss)

                entropy_loss = None
                if self.entropy_optimizer is not None:
                    with torch.autocast(
                        device_type=self._device_type, enabled=self.cfg.mixed_precision
                    ):
                        entropy_loss = -(
                            self.log_entropy_coefficient
                            * (log_prob + self._target_entropy).detach()
                        ).mean()
                    self.entropy_optimizer.zero_grad()
                    self.scaler.scale(entropy_loss).backward()
                    self.scaler.step(self.entropy_optimizer)
                    self._entropy_coefficient = self.log_entropy_coefficient.detach().exp()

                if self.cfg.deterministic_policy and self._q_method == "bellman":
                    self.target_policy.update_parameters(self.policy, polyak=self.cfg.polyak)
                if self._q_method == "bellman":
                    self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
                    self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)

                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Policy update / Q (mean)", policy_q.mean().item())
                if bc_loss is not None:
                    self.track_data("Loss / BC loss", bc_loss.item())
                    self.track_data("Coefficient / DDPG+BC lambda", bc_lambda.item())
                if log_prob is not None:
                    self.track_data("Policy / log_prob (mean)", log_prob.mean().item())
                    self.track_data(
                        "Entropy / Entropy coefficient", float(self._entropy_coefficient)
                    )
                if entropy_loss is not None:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())

            self.scaler.update()
            if self.policy_scheduler is not None:
                self.policy_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()

            self.track_data("Loss / Critic loss", critic_loss.item())
            if value_loss is not None:
                self.track_data("Loss / Value loss", value_loss.item())
                self.track_data("IQL / V (mean)", values.mean().item())
            self.track_data("Q-network / Q1 (mean)", critic_1_values.mean().item())
            self.track_data("Q-network / Q2 (mean)", critic_2_values.mean().item())
            self.track_data("Target / Target (mean)", target_values.mean().item())
