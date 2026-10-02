from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from skrl import config, logger
# from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer
# from skrl.utils import postprocessing

from algorithms import postprocessing
from .osrl_cfg import OSRL_CFG
from .base_c import Agent


class OSRL(Agent):
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
        cfg: OSRL_CFG | dict = {},
    ) -> None:
        """One-Shot RL (OSRL).

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
        self.cfg: OSRL_CFG
        super().__init__(
            models=models,
            models_ref=models_ref,
            memory=memory,
            expert_memory=expert_memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=OSRL_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        self._DeepEnsemble = self.cfg.Deep_Ensemble
        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        self.target_policy = self.models.get("target_policy", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)


        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2


        self.critics = []
        if self._DeepEnsemble:
            self.critics = [
                self.models.get(f"critic_{i+1}", None)
                for i in range(self.cfg.Ensemble_size)
            ]
        else:
            self.critics = [
                self.models.get("critic_1", None),
                self.models.get("critic_2", None),
            ] 

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"]    = self.value

        for i, critic in enumerate(self.critics):
            self.checkpoint_modules[f"critic_{i+1}"] = critic


        # IQL parameters
        self._expectile = self.cfg.expectile
        self._temperature = self.cfg.temperature

        # Demo file
        self._offline_mode = self.cfg.offline_mode
        self._demo_file = self.cfg.demo_file       

        # Experiment module
        self._select_method = self.cfg.select_method
        self._policy_method = self.cfg.policy_method
        self._q_method = self.cfg.q_method
        self._UAQ = self.cfg.UAQ
        self.Need_new_replay = False
        self.pre_train = False
        self.bc_ = 0.0
        self.q_gain = 0.0

        # Replay demo
        self.Ref_policy = self.models_ref.get("policy", None)

        self._num_envs = self.cfg.num_envs

        self.episode_done = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self.epi_timestep = 0
        self._cached_exp_observations: torch.Tensor | None = None
        self._cached_exp_actions: torch.Tensor | None = None

        # Caculate state OOD
        self.demo_mean = 0.0
        self.demo_cov = 0.
        self.demo_mean0 = 0
        self.bias = torch.Tensor([0.])
        self.il_selected_rate = 0.
        self.return_val = torch.zeros(self._num_envs, device=self.device)
        self._episode_buffer = [[] for _ in range(self._num_envs)]
        self.replay_re = False
        self.max_return = 100.
        self.use_rl = torch.zeros(self._num_envs, 1, device=self.device)
        
        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            all_models = [self.policy, self.value] + self.critics
            for m in all_models:
                if m is not None:
                    m.broadcast_parameters()


        # set up noise
        # - exploration noise
        if self.cfg.exploration_noise is not None:
            self._exploration_noise = self.cfg.exploration_noise(**self.cfg.exploration_noise_kwargs)
        else:
            logger.warning("agents:OSRL: No exploration noise specified, training performance may be degraded")
            self._exploration_noise = None
        # - smooth regularization noise
        if self.cfg.smooth_regularization_noise is not None:
            self._smooth_regularization_noise = self.cfg.smooth_regularization_noise(
                **self.cfg.smooth_regularization_noise_kwargs
            )
        else:
            logger.warning("agents:OSRL: No smooth regularization noise specified, training variance may be high")
            self._smooth_regularization_noise = None

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # optimizers
        if self.policy is not None and self.critics and self.value is not None:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self.cfg.learning_rate[0]
            )
            # [NEW] Single optimizer chains ALL ensemble critics.
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(*[c.parameters() for c in self.critics]),
                lr=self.cfg.learning_rate[0],
            )
            self.value_optimizer = torch.optim.Adam(
                self.value.parameters(), lr=self.cfg.learning_rate[0]
            )
 
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
            self.checkpoint_modules["value_optimizer"]  = self.value_optimizer
            # - learning rate schedulers
            # schedulers  (index 0 → policy, 1 → critic, 2 → value)
            def _make_scheduler(idx, optimizer):
                cls = self.cfg.learning_rate_scheduler[idx]
                if cls is None:
                    return None
                return cls(optimizer, **self.cfg.learning_rate_scheduler_kwargs[0])
 
            self.policy_scheduler  = _make_scheduler(0, self.policy_optimizer)
            self.critic_scheduler  = _make_scheduler(1, self.critic_optimizer)
            self.value_scheduler   = _make_scheduler(1, self.value_optimizer)

        # set up target networks
        if self.target_policy is not None and self.target_critic_1 is not None and self.target_critic_2 is not None:
            # - freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)
            # - update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic_1.update_parameters(self.critics[0], polyak=1)
            self.target_critic_2.update_parameters(self.critics[1], polyak=1)


        # entropy
        self._entropy_coefficient = self.cfg.initial_entropy_value
        if self.cfg.learn_entropy:
            self._target_entropy = self.cfg.target_entropy
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self.cfg.learning_rate[1])
            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer


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

    # ------------------------------------------------------------------
    # Helper: compute discounted return-to-go for a single trajectory
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_mc_returns(rewards: list, terminated: list, discount: float) -> list:
        """Compute discounted Monte-Carlo return-to-go for one trajectory.

        Args:
            rewards:    list of scalar rewards, length T.
            terminated: list of done flags (bool/float), length T.
            discount:   discount factor γ.

        Returns:
            List of return-to-go values of the same length T.
        """
        mc_returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(terminated)):
            if d[0]:          # episode ended → reset accumulator
                G = 0.0
            G = float(r[0]) + discount * G
            mc_returns.append(G)
        mc_returns.reverse()
        return mc_returns
    
    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="useRL", size=1, dtype=torch.float32)

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



        # create tensors in memory
        # # self.expert_memory
        if self.expert_memory is not None:
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
            
            G = 0
            for j, data in expert_memory:
                keys = list(data.keys())
                N = len(data[keys[0]])
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

            expert_states_r = self.expert_memory.sample(names=["observations"],
                                                        batch_size=N)[0][0]
            self.demo_mean0 = torch.mean(expert_states_r, dim=0)
            cov = torch.cov(expert_states_r.T)
            self.demo_cov = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0], device= cov.device))
            self._refresh_expert_replay_cache()


        # Todo need to be done in the task.
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)
            self.action_scale = (self.clip_actions_max - self.clip_actions_min)/2
            self.action_bias = (self.clip_actions_min + self.clip_actions_max)/2

        print('self.clip_actions_min', self.clip_actions_min)
        print('self.clip_actions_max', self.clip_actions_max)

        # create temporary variables needed for storage and computation
        self._update_counter = 0

    def _refresh_expert_replay_cache(self) -> None:
        """Cache the expert trajectory used by ``act`` until expert memory changes."""
        if self.expert_memory is None:
            raise RuntimeError("Expert memory is required for replay action selection")
        sampled_exp_observations, sampled_exp_actions = self.expert_memory.sample_all(
            names=["observations", "actions"],
            mini_batches=1,
            sequence_length=self.cfg.episode_steps,
        )[0]
        self._cached_exp_observations = sampled_exp_observations.detach().clone()
        self._cached_exp_actions = sampled_exp_actions.detach().clone()


    def _select_act(self, obs: torch.Tensor,
        exp_obs: torch.Tensor,
        rl_actions: torch.Tensor,
        replay_actions: torch.Tensor,
        soft: bool,
        target: bool,):

        inputs = {
            "observations": self._observation_preprocessor(obs),
            "states": self._state_preprocessor(obs),
        }

        exp_inputs = {
            "observations": self._observation_preprocessor(exp_obs),
            "states":self._state_preprocessor(exp_obs),
        }

        # Stack actions and get batch dimensions
        rl_bc_actions = torch.stack([rl_actions, replay_actions], dim=1)
        num_envs, num_action, _ = rl_bc_actions.size()  # get dimensions values, bsize:batch size

        if self._select_method == "iql_v":
            # Compute values for both states
            v_pi, _  = self.value.act(inputs, role="value")
            v_pi_exp, _  = self.value.act(exp_inputs, role="value")
            # Stack values
            values = torch.stack([v_pi, v_pi_exp], dim=1).view(num_envs, num_action)
            # Logging
            self.track_data("Select / Policy Value(s) mean", v_pi.mean().item())
            self.track_data("Select / Replay Value(s') mean", v_pi_exp.mean().item())         
        elif self._select_method == "off_state":
            # Compute values for both states
            _, q_pi, q_std_pi = self._ensemble_q(inputs, rl_actions)
            _, q_pi_exp, q_std_pi_exp = self._ensemble_q(exp_inputs, replay_actions)
            # q_pi_rel = q_pi 
            # q_pi_exp_rel = q_pi_exp 
            q_pi_rel = q_pi * torch.exp(-q_std_pi)
            q_pi_exp_rel = q_pi_exp * torch.exp(-q_std_pi_exp)
            values = torch.stack([q_pi_rel, q_pi_exp_rel], dim=1).view(num_envs, num_action)
            self.track_data("Select / Policy Q(s, a) mean", torch.mean(q_pi_rel).item())
            self.track_data("Select / Replay Q(s', a') mean", torch.mean(q_pi_exp_rel).item())
            self.track_data("Select / Policy Q(s, a) std", torch.mean(q_std_pi).item())
            self.track_data("Select / Replay Q(s', a') std", torch.mean(q_std_pi_exp).item())

        elif self._select_method == "adv":
            with torch.no_grad():
                BC_dist = F.mse_loss(obs, exp_obs)
                state_tune = BC_dist.abs().mean().detach()
    
            self.track_data("Loss / states BC dis", torch.mean(BC_dist).item())
            v_pi, _  = self.value.act(inputs, role="value")
            # Compute Q-values for both states
            _, q_pi, q_std_pi = self._ensemble_q(inputs, rl_actions)
            _, q_pi_exp, q_std_pi_exp = self._ensemble_q(inputs, replay_actions)
            # q_pi_rel = q_pi 
            # q_pi_exp_rel = q_pi_exp 
            if self.cfg.uncertainty_epsilon > 0.:
                q_pi_rel = q_pi 
                q_pi_exp_rel = q_pi_exp
            else:
                q_pi_rel = q_pi * torch.exp(-q_std_pi)
                q_pi_exp_rel = q_pi_exp * torch.exp(-q_std_pi_exp)
            adv_rl = q_pi_rel - v_pi
            adv_replay = q_pi_exp_rel -v_pi
            values = torch.stack([adv_rl, adv_replay], dim=1).view(num_envs, num_action)
            self.track_data("Select / Policy Q(s, a) mean", torch.mean(adv_rl).item())
            self.track_data("Select / Replay Q(s, a') mean", torch.mean(adv_replay).item())
            self.track_data("Select / Policy Q(s, a) std", torch.mean(q_std_pi).item())
            self.track_data("Select / Replay Q(s, a') std", torch.mean(q_std_pi_exp).item())

        else:  
            with torch.no_grad():
                BC_dist = F.mse_loss(obs, exp_obs)  
            self.track_data("Loss / states BC dis", torch.mean(BC_dist).item())
            # Compute values for both states
            _, q_pi, q_std_pi = self._ensemble_q(inputs, rl_actions)
            _, q_pi_exp, q_std_pi_exp = self._ensemble_q(inputs, replay_actions)
            # if self.cfg.uncertainty_epsilon > 0.:
            #     q_pi_rel = q_pi 
            #     q_pi_exp_rel = q_pi_exp
            # else:
            #     q_pi_rel = q_pi * torch.exp(-q_std_pi)
            #     q_pi_exp_rel = q_pi_exp * torch.exp(-q_std_pi_exp)
            q_pi_rel = q_pi 
            q_pi_exp_rel = q_pi_exp 
            # if self.cfg.UAQ:
            #     with torch.no_grad():
            #         state_tune = BC_dist.abs().mean().detach()
            #         # BC_log_prob = self.policy.evaluate(exp_inputs, replay_actions)
            #         BC_log_prob = self.policy.evaluate(inputs, rl_actions)
            #     bc_ = max(-BC_log_prob.mean().detach().item(), 0.0)
            #     q_pi_rel = q_pi + bc_*torch.exp(-1*q_std_pi)
            #     q_pi_exp_rel = q_pi_exp + bc_*torch.exp(-1*q_std_pi_exp)
            values = torch.stack([q_pi_rel, q_pi_exp_rel], dim=1).view(num_envs, num_action)
            self.track_data("Select / Policy Q(s, a) mean", torch.mean(q_pi_rel).item())
            self.track_data("Select / Replay Q(s, a') mean", torch.mean(q_pi_exp_rel).item())
            self.track_data("Select / Policy Q(s, a) std", torch.mean(q_std_pi).item())
            self.track_data("Select / Replay Q(s, a') std", torch.mean(q_std_pi_exp).item())
            self.track_data("Select / Raw Policy Q(s, a) mean", torch.mean(q_pi).item())
            self.track_data("Select / Raw Replay Q(s, a') mean", torch.mean(q_pi_exp).item())
            


        # Select actions based on strategy
        if soft:
            # Boltzmann exploration
            # probs = F.softmax(target_q_values * self._soft_update_beta, dim=1)
            probs = torch.nn.functional.softmax(values, dim=1)
            action_indices = probs.multinomial(1)
            use_bc = action_indices.sum().item()
            # print("probs", probs)
            # ref_actions,_ = self.Ref_policy.act(inputs, role="policy")
            actions = rl_actions * (1 - action_indices) + replay_actions * action_indices
        else:
            # Greedy selection
            action_indices = values.argmax(dim=1)
            use_bc = action_indices.sum().item()
            
            # print("action_indices", action_indices)
            # print("use_bc", use_bc)
            actions = rl_bc_actions[torch.arange(num_envs), action_indices]
        if not target:
            self.use_rl = (1 - action_indices).float().unsqueeze(-1) 
            Replay_BC = torch.norm(actions - replay_actions, dim=1)
            
            self.track_data("Interact / interaction distance", Replay_BC.mean().item())
            self.track_data("Select / selected_action_index", (1- use_bc/self._num_envs))
            # self.track_data("Select / adaptive weight", bc_)

            _, q_pi_replay, q_std_replay = self._ensemble_q(exp_inputs, replay_actions)
            if self.cfg.uncertainty_epsilon > 0.:
                q_pi_replay = q_pi_replay
            else:
                q_pi_replay = q_pi_replay * torch.exp(-q_std_replay)
            self.track_data("Select / Original Replay Q(s', a') mean", torch.mean(q_pi_replay).item())
        return actions
      
    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(observations),
        }

        # Evaluation is policy-only: do not replay the demonstration, add
        # exploration noise, or run the IBRL policy/demo action selector. For
        # a stochastic policy, SequentialTrainer executes ``mean_actions`` by
        # default and can execute the sampled action with stochastic_evaluation.
        if not self.training:
            with torch.autocast(
                device_type=self._device_type, enabled=self.cfg.mixed_precision
            ):
                actions, outputs = self.policy.act(inputs, role="policy")
            actions = actions.clamp(min=self.clip_actions_min, max=self.clip_actions_max)
            outputs = dict(outputs)
            mean_actions = outputs.get("mean_actions", actions)
            outputs["mean_actions"] = mean_actions.clamp(
                min=self.clip_actions_min, max=self.clip_actions_max
            )
            return actions, outputs

        if self._cached_exp_observations is None or self._cached_exp_actions is None:
            raise RuntimeError("Expert replay cache was not initialized")
        sampled_exp_observations = self._cached_exp_observations
        sampled_exp_actions = self._cached_exp_actions

        # reset to 0 if any env is done, otherwise increment
        if self.episode_done.any():
            self.epi_timestep = 0
        else:
            self.epi_timestep += 1
        if self.epi_timestep >= sampled_exp_actions.shape[0]:
            self.epi_timestep = sampled_exp_actions.shape[0] - 1
        current_exp_observation = sampled_exp_observations[self.epi_timestep-1, :].repeat(self._num_envs, 1)
        current_exp_action = sampled_exp_actions[self.epi_timestep-1, :].repeat(self._num_envs, 1)



        #  For detect state OOD 
        diff = observations - self.demo_mean0
        left = torch.matmul(diff, self.demo_cov)
        dist_sq = (left*diff).sum(dim=1)
        M_dist = torch.sqrt(dist_sq)

        self.track_data("Loss / states BC loss", torch.mean(M_dist).item())
        
        # sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):

            if timestep < self.cfg.learning_starts:
                actions = current_exp_action
                outputs = {}
                action_idx = 0
                if self._exploration_noise:
                    noises = self._exploration_noise.sample(actions.shape)
                    if self.cfg.exploration_scheduler:
                        noises.mul_(self.cfg.exploration_scheduler(timestep, timesteps))
                        actions.add_(noises)
                    actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                    self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
                    self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
                    self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item()) 
                return actions, outputs 
            
            # IBRL action selection
            ## pretrain
            actions, outputs = self.policy.act(inputs, role="policy")
            exp_inputs = {
            "observations": self._observation_preprocessor(current_exp_observation),
            "states":self._state_preprocessor(current_exp_observation),
        }
            BC_log_prob = self.policy.evaluate(exp_inputs, current_exp_action)


            # recording L2 distance for analysis
            L2_dis = torch.norm(actions - current_exp_action, dim=1)
            self.track_data("Interact / L2 distance", L2_dis.mean().item())
            self.track_data("Interact / BC log prob", BC_log_prob.mean().item())


            if not self.cfg.deterministic_policy:
                log_std = outputs["log_std"]
                self.track_data("Policy / log_std (mean)", log_std.mean().item())
                self.track_data("Policy / std (mean)", log_std.exp().mean().item())

            if self.cfg.deterministic_policy:
                if self._exploration_noise:
                    noises = self._exploration_noise.sample(actions.shape)
                    if self.cfg.exploration_scheduler:
                        noises.mul_(self.cfg.exploration_scheduler(timestep, timesteps))
                    if  self.replay_re:
                        noises = noises.mul_(0.0001)
                        actions.add_(noises)
                    else:
                        actions.add_(noises)
                    actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                    self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
                    self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
                    self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item()) 

    
            if self.cfg.select_act:
                ## select actions
                actions = self._select_act(observations, current_exp_observation, actions, current_exp_action, soft=False, target=False)
                


            if timestep < self.cfg.random_timesteps:
                actions = current_exp_action
                outputs = {}

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
        if not self.training:
            # Evaluation records episode statistics in the base agent only.
            # It must not alter online replay, expert replay, or policy-selection state.
            self.episode_done = (terminated | truncated).reshape(-1).bool()
            return

        epi_timestep = timestep % self.cfg.episode_steps
        if epi_timestep != 0:
            self.return_val = self.return_val + rewards.squeeze(-1)
        else:
            self.return_val = torch.zeros_like(self.return_val)

        
        if self.memory is not None:
            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                useRL =self.use_rl,
            )


        if timestep < self.cfg.learning_starts:
            return
        
        if timestep == timesteps - 1:
            self.memory.save("demos", format="csv")
        
        # buffer the transition instead of storing immediately
        for i in range(self._num_envs):
            self._episode_buffer[i].append(dict(
                observations=observations[i],
                states=observations[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_observations[i],
                next_states=next_observations[i],
                next_actions=torch.zeros_like(actions[i]),  # placeholder
                terminated=terminated[i],
                truncated=truncated[i],
            ))

            # fix next_actions in buffer: t.next_action = t+1.action
            if len(self._episode_buffer[i]) >= 2:
                self._episode_buffer[i][-2]["next_actions"] = actions[i].clone()

        # at episode end, flush buffer to memory only if return qualifies
        self.episode_done = (terminated | truncated) 

        if epi_timestep == (self.cfg.episode_steps - 1) or self.episode_done.any():
            self.track_data("Return / return mean ", torch.mean(self.return_val).item())
            if not self.pre_train:
                print("self.pre_train = False")
            
            # if (self.return_val > self.cfg.suc_score).any():    #120
            # # if torch.mean(self.return_val).item() > 100:
            #     self.replay_re = True
            #     self.Need_new_replay = False
            #     # print("self.replay_re = True")
            # else:
            #     self.replay_re = False
            #     self.Need_new_replay = True
            #     self.max_return = self.cfg.suc_score
            #     # print(f"self.Need_new_replay = True, reset self.max_return to {self.cfg.suc_score}")
            

  
        expert_memory_changed = False
        for i in range(self._num_envs):
            if self.episode_done[i] or epi_timestep == (self.cfg.episode_steps -1):
                env_return = self.return_val[i].item()

                # if torch.mean(self.return_val).item() > self.max_return:
                if env_return > self.max_return:
                    for transition in self._episode_buffer[i]:
                        self.expert_memory.add_samples(**transition)
                    expert_memory_changed = True
                    # self._q_method = "iql"
                    # self._policy_method = "DDPG+BC"
                    self.max_return = 1200
                    print("recording new replay for lacking for initial reward tasks")
                if timestep <= self.cfg.random_timesteps - 1:
                    for transition in self._episode_buffer[i]:
                        self.expert_memory.add_samples(**transition)
                    expert_memory_changed = True
                    self.cfg.random_timesteps = 0
                    print("recording replay")

                if env_return > self.max_return:
                    self.pre_train = True
                    self.pretrain_step = timestep
                    self.max_return = env_return
                    print(f"New best env {i} with return {env_return}")


                # clear only this env
                self._episode_buffer[i] = []
                self.return_val[i] = 0.0
                # self.max_return = 0.

        if expert_memory_changed:
            self._refresh_expert_replay_cache()

        if self.pre_train:
            if timestep > self.pretrain_step + self.cfg.random_timesteps or self.Need_new_replay:
                self.pre_train = False
                

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


    def expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """Asymmetric L2 loss used to train the value network.
    
        For diff = Q(s,a) - V(s):
        - weight = τ     when diff > 0  (over-estimate region)
        - weight = 1 - τ when diff ≤ 0  (under-estimate region)
    
        Reference: eq. (4) in https://arxiv.org/abs/2110.06169
        """
        weight = torch.where(diff > 0,
                            torch.full_like(diff, expectile),
                            torch.full_like(diff, 1.0 - expectile))
        return (weight * diff.pow(2)).mean()

 

    def _ensemble_q(
        self,
        inputs: dict,
        actions: torch.Tensor,
        no_grad: bool = False,
        policy_extraction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query all ensemble critics and return stacked values plus statistics.
 
        Returns
        -------
        q_stack : (N, B, 1)   raw per-critic values
        q_mean  : (B, 1)      mean over ensemble
        q_std   : (B, 1)      std  over ensemble  ← uncertainty estimate
        """
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            q_list = [
                c.act({**inputs, "taken_actions": actions}, role=f"critic_{i+1}")[0]
                for i, c in enumerate(self.critics)
            ]
        q_stack = torch.stack(q_list, dim=0)         # (N, B, 1)
        q_std   = q_stack.std(dim=0)                 # (B, 1)  uncertainty

        if self._DeepEnsemble:
            q_mean  = q_stack.mean(dim=0)                # (B, 1)
        else:
            q_mean, _ = q_stack.min(dim=0)                # (B, 1)  conservative estimate

        ## new*******************## new*******************
        if self.cfg.UAQ:
            if policy_extraction:
                with torch.no_grad():
                    BC_log_prob = self.policy.evaluate(inputs, actions)
                    # BC_log_prob = self.policy.evaluate(exp_inputs, sampled_exp_actions)
                    self.bc_ = max(-BC_log_prob.mean().detach().item(), 1.0)
                    self.q_gain = max(q_mean.mean().detach().item(), 1.0)
            else:
                # q_pi = q_pi*torch.exp(-bc_*q_std_pi * torch.sign(q_pi))
                # q_mean = q_mean *torch.exp(-q_std)
                # q_mean = q_mean + self.bc_*torch.exp(-self.q_gain*q_std)
                q_mean = q_mean + torch.exp(-q_std)
                self.track_data("Policy update / adaptive weight", self.bc_)
                self.track_data("Policy update / q_pi", torch.mean(q_mean).item())

        return q_stack, q_mean, q_std

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # if not self.cfg.select_method and self._q_method != "iql":
        if not self.cfg.select_act:
            batch_size = int(self.cfg.batch_size/2)
        else:
            batch_size = self.cfg.batch_size

        # sample a batch from memory
        (
            sampled_observations,
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_next_observations,
            sampled_next_states,
            sampled_terminated,
            sampled_truncated,
        ) = self.memory.sample(names=self._tensors_names, batch_size=batch_size)[0]


        # sample a batch from expert memory
        (
            sampled_exp_observations,
            sampled_exp_states,
            sampled_exp_actions,
            sampled_exp_rewards,
            sampled_exp_next_observations,
            sampled_exp_next_states,
            sampled_exp_next_actions,
            sampled_exp_terminated,
            sampled_exp_truncated,
        ) = self.expert_memory.sample(names=self._tensors_names_exp, batch_size=batch_size)[0]

        # if not self.cfg.select_method and self._q_method != "iql":
        if not self.cfg.select_act and timestep > self.cfg.random_timesteps:
            sampled_observations = torch.cat([sampled_observations, sampled_exp_observations], dim=0)
            # sampled_states = torch.cat([sampled_observations, sampled_exp_observations], dim=0)
            sampled_actions = torch.cat([sampled_actions, sampled_exp_actions], dim=0)
            sampled_rewards = torch.cat([sampled_rewards, sampled_exp_rewards], dim=0)
            sampled_next_observations = torch.cat([sampled_next_observations, sampled_exp_next_observations], dim=0)
            # sampled_next_states = torch.cat([sampled_next_observations, sampled_exp_next_observations], dim=0)
            sampled_terminated = torch.cat([sampled_terminated, sampled_exp_terminated], dim=0)
            sampled_truncated = torch.cat([sampled_truncated, sampled_exp_truncated], dim=0)


        # gradient steps
        for gradient_step in range(self.cfg.gradient_steps):
            
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                    "states": self._state_preprocessor(sampled_observations, train=True),
                }
                exp_inputs = {
                    "observations": self._observation_preprocessor(sampled_exp_observations, train=True),
                    "states": self._state_preprocessor(sampled_exp_observations, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                    "states": self._state_preprocessor(sampled_next_observations, train=True),
                }
                if self._q_method == "iql":
                # ── 1. Value update ───────────────────
                    with torch.no_grad():
                        # No target policy smoothing, as IQL only queries from in-sample learning.
                        # Directly compute in-sample critic value 
                        q_stack, q_mean, _ = self._ensemble_q(inputs, sampled_actions, no_grad=True)
                        # q_stack: (N, B, 1) → min over ensemble dim → (B, 1)
                        critic_value = q_stack.min(dim=0).values
                        # critic_value = q_stack.mean()


                    v_values, _ = self.value.act(inputs, role="value")          # V(s)
                    value_loss  = self.expectile_loss(critic_value - v_values, self._expectile)

                    self.value_optimizer.zero_grad()
                    self.scaler.scale(value_loss).backward()
                    if self.cfg.grad_norm_clip > 0:
                        self.scaler.unscale_(self.value_optimizer)
                        nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_norm_clip)
                    self.scaler.step(self.value_optimizer)

            ## ── 2. Critic update ───────────────────
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                # compute taegrt value based on value function estimation on sampled next states (baseline value).
                with torch.no_grad():
                    if self._q_method == "iql":
                        # v_next, select_rl = self._select_rl(sampled_next_observations, sampled_exp_next_observations, sampled_exp_actions)
                        v_next, _ = self.value.act(next_inputs, role="value")   # V(s')
                        q_backup  = sampled_rewards + self.cfg.discount_factor * (sampled_terminated | sampled_truncated).logical_not() * v_next
                    else:
                        if self.cfg.deterministic_policy:
                            # target policy smoothing
                            next_actions, next_outputs = self.target_policy.act(next_inputs, role="target_policy")   


                            ## select actions
                            if self.cfg.select_act:
                                next_actions = self._select_act(sampled_next_observations, sampled_exp_next_observations, next_actions, sampled_exp_next_actions, soft=False, target=True)

                            if self._smooth_regularization_noise is not None:
                                noises = torch.clamp(
                                    self._smooth_regularization_noise.sample(next_actions.shape),
                                    min=-self.cfg.smooth_regularization_clip,
                                    max=self.cfg.smooth_regularization_clip,
                                )
                                next_actions.add_(noises)
                                next_actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)
                        else:
                            if self._q_method == "cql":
                                # Random actions (uniform coverage of action space)
                                rand_actions, _ = self.policy.random_act(inputs, role="policy")
                                # Current policy on current observations
                                cur_actions, _ = self.policy.act(inputs, role="policy")

                            next_actions, next_outputs = self.policy.act(next_inputs, role="target_policy")

                            ## select actions
                            if self.cfg.select_act:
                                next_actions = self._select_act(sampled_next_observations, sampled_exp_next_observations, next_actions, sampled_exp_next_actions, soft=False, target=True)


                        # compute target values
                        target_q1_values, _ = self.target_critic_1.act(
                            {**next_inputs, "taken_actions": next_actions}, role="target_critic_1"
                        )
                        target_q2_values, _ = self.target_critic_2.act(
                            {**next_inputs, "taken_actions": next_actions}, role="target_critic_2"
                        )
                        if self.cfg.deterministic_policy:
                            # Standard DDPG-style backup: min double Q.
                            target_q_values = torch.min(target_q1_values, target_q2_values)
                        else:
                            # Max-entropy backup: min double Q - α * log π.
                            log_prob = next_outputs["log_prob"]
                            if self.cfg.learn_entropy:
                                target_q_values = torch.min(target_q1_values, target_q2_values) - self._entropy_coefficient * log_prob
                            else:
                                target_q_values = torch.min(target_q1_values, target_q2_values)

                        target_values = (
                            sampled_rewards
                            + self.cfg.discount_factor
                            * (sampled_terminated | sampled_truncated).logical_not()
                            * target_q_values
                        )

                critic_loss = torch.tensor(0.0, device=self.device)
                q_values_list = []
                for i, critic in enumerate(self.critics):
                    q_i, _ = critic.act(
                        {**inputs, "taken_actions": sampled_actions}, role=f"critic_{i+1}"
                    )
                    q_values_list.append(q_i)
                    if self._q_method == "iql":
                        critic_loss = critic_loss + F.mse_loss(q_i, q_backup)
                    elif self._q_method == "cql":
                        # ── OOD Q-values for logsumexp penalty ──────────────────────────
                        q_rand, _    = critic.act({**inputs, "taken_actions": rand_actions},  role=f"critic_{i+1}")
                        q_cur,  _    = critic.act({**inputs, "taken_actions": cur_actions},   role=f"critic_{i+1}")
                        q_next, _    = critic.act({**inputs, "taken_actions": next_actions},   role=f"critic_{i+1}")


                        # cat shape: (B, 3) — rand | cur-policy | next-policy
                        cat_q = torch.cat([q_rand, q_cur, q_next], dim=1)
                        n_actions = cat_q.shape[1]

                        # ── CQL conservative penalty ────────────────────────────────────
                        cql_logsumexp = (
                            torch.logsumexp(cat_q / self.cfg.cql_temperature, dim=1)
                            - math.log(n_actions)
                        )

                        cql_loss = (
                            self.cfg.cql_alpha * self.cfg.cql_temperature * cql_logsumexp.mean()
                            - self.cfg.cql_alpha * q_i.mean()
                        )

                        critic_loss = critic_loss + F.mse_loss(q_i, target_values) + cql_loss
                    else:
                        critic_loss = critic_loss + F.mse_loss(q_i, target_values)

                # Recompute summary statistics (with grad for logging).
                q_stack_train = torch.stack(q_values_list, dim=0)   # (N, B, 1)
                # Ensemble uncertainty on dataset actions (training batch).
                q_mean_train  = q_stack_train.mean(dim=0)
                q_std_train   = q_stack_train.std(dim=0)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                for critic in self.critics:
                    critic.reduce_parameters()
 
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(*[c.parameters() for c in self.critics]),
                    self.cfg.grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)

            # # ── 3. Policy update (every policy_delay steps) ───────────────────
            self._update_counter += 1
            if not self._update_counter % self.cfg.policy_delay:

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    # Advantage-weighted regression (AWR)
                    # ─────────────────────────────────────
                    # The policy is NOT used to compute advantage targets, so
                    # it is only updated on *in-distribution* (dataset) actions.
                    #
                    # A(s,a) = Q(s,a) - V(s)   [no gradient through Q/V]
                    # w(s,a) = clamp(exp(β · A(s,a)), max=EXP_ADV_MAX)
                    # L_π    = -mean( w · log π(a|s) )
                    
                    if self._policy_method == "DDPG+BC":
                        # 1. Q maximization term: Q(s, μ(s))
                        actions, outputs = self.policy.act(inputs, role="policy")
                        # Use ensemble mean Q for the Q-maximisation term.
                        _, q_pi, q_std_pi = self._ensemble_q(inputs, actions, policy_extraction=True)
                        self.track_data("Policy update / q_pi raw", torch.mean(q_pi).item())

                        q_loss = -q_pi.mean()   # maximize Q
                        if self.cfg.deterministic_policy:
                            # Standard DDPG+BC uses MSE
                            BC_loss = F.mse_loss(actions, sampled_actions)
                            alpha = 0.1
                            lambda_ = alpha / q_pi.abs().mean().detach()
                            policy_loss = lambda_*(q_loss)+ BC_loss
                        else:
                            # Normalise dataset actions back to tanh space
                            # Evaluate log-prob of dataset actions under current policy
                            log_prob = self.policy.evaluate(inputs, sampled_actions)

                            log_prob_pi = outputs["log_prob"]
                            entropy_loss = log_prob_pi.mean()          # minimizing this = maximizing entropy(randomness) of policy actions
                            gamma_ = 1/log_prob_pi.abs().mean().detach()

                            gamma_2 = 1/log_prob.abs().mean().detach()
                            # gamma_2 = log_prob.abs().mean().detach()

                            log_std_pi = outputs["log_std"]
                            gamma_3 = 1*log_std_pi.exp().mean().detach()
                            # self.track_data("Policy / log_std_pi.exp().mean()", log_std_pi.exp().mean())


                            BC_loss = -log_prob
                            alpha = 1
                            lambda_ = alpha / q_pi.abs().mean().detach()
                            if self.cfg.learn_entropy:
                                policy_loss = lambda_*(q_loss)+ gamma_3*BC_loss.mean() + self._entropy_coefficient * entropy_loss
                            else:
                                policy_loss = lambda_*q_loss+ gamma_3*BC_loss.mean()
                    elif self._policy_method == "AWR":
                        with torch.no_grad():
                            q_stack, q_mean, q_std_pi = self._ensemble_q(inputs, sampled_actions, no_grad=True)
                            q_pi = q_stack.min(dim=0).values
                            # q_pi = q_stack.mean()

                            # if select_rl:
                            v_pi, _  = self.value.act(inputs, role="value")
                            advantage = q_pi - v_pi                                 # A(s,a)
                            exp_adv   = torch.exp(self._temperature * advantage).clamp(
                                max=self.cfg.exp_adv_max
                            ) 
                        if self.cfg.deterministic_policy:
                            # compute policy (actor) loss           
                            actions, _ = self.policy.act(inputs, role="policy")
                            # ── weighted regression loss ─────────────────────────────────────
                            # MSE per sample (sum over action dims)
                            BC_loss = (actions - sampled_actions).pow(2).sum(dim=-1, keepdim=True)
                            policy_loss = (exp_adv * BC_loss).mean()
                        else:
                            # Normalise dataset actions back to tanh space
                            # Evaluate log-prob of dataset actions under current policy
                            log_prob = self.policy.evaluate(inputs, sampled_actions)
                            actions, outputs = self.policy.act(inputs, role="policy")
                            if self.cfg.UAQ:
                                _, q_pi, q_std_pi = self._ensemble_q(inputs, actions, policy_extraction=True)
                            log_prob_pi = outputs["log_prob"]
                            entropy_loss = log_prob_pi.mean()          # minimizing this = maximizing entropy(randomness) of policy actions
        
                            BC_loss = -log_prob
                            AWR_loss = (exp_adv * BC_loss).mean()

                            if self.cfg.learn_entropy:
                                policy_loss = AWR_loss + self._entropy_coefficient * entropy_loss
                            else:
                                policy_loss = AWR_loss
                    else:
                        actions, outputs = self.policy.act(inputs, role="policy")
                        BC_loss = F.mse_loss(actions, sampled_actions)
                        _, q_pi, q_std_pi = self._ensemble_q(inputs, actions)
                        # q_pi = q_pi * torch.exp(-2 * q_std_pi.mean())  ## new
                        if self.cfg.deterministic_policy:
                            policy_loss = -q_pi.mean()
                        else:
                            log_prob = outputs["log_prob"]
                            log_prob_pi = outputs["log_prob"]
                            entropy_loss = log_prob.mean()          # minimizing this = maximizing entropy(randomness) of policy actions
                            if self.cfg.learn_entropy:
                                policy_loss = -q_pi.mean() + self._entropy_coefficient * entropy_loss
                            else:
                                policy_loss = -q_pi.mean()

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)

                self.scaler.step(self.policy_optimizer)

                # entropy learning
                if not self.cfg.deterministic_policy:
                    if self.cfg.learn_entropy:
                        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                            # compute entropy loss
                            entropy_loss = -(self.log_entropy_coefficient * (log_prob_pi).detach()).mean()

                        # optimization step (entropy)
                        self.entropy_optimizer.zero_grad()
                        self.scaler.scale(entropy_loss).backward()
                        self.scaler.step(self.entropy_optimizer)

                        # compute entropy coefficient
                        self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())
                if self.cfg.deterministic_policy:
                    self.target_policy.update_parameters(self.policy, polyak=self.cfg.polyak) 
                if self._q_method != "iql":       
                    # update target networks
                    self.target_critic_1.update_parameters(self.critics[0], polyak=self.cfg.polyak)
                    self.target_critic_2.update_parameters(self.critics[1], polyak=self.cfg.polyak)



            self.scaler.update()  # called once, after optimizers have been stepped


            # update learning rate
            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            # record data
            if not self._update_counter % self.cfg.policy_delay:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / BC loss", BC_loss.mean().item())
                if not self.cfg.deterministic_policy:
                    self.track_data("Policy / log_prob (mean)", log_prob.mean().item())
                    if self.cfg.learn_entropy:
                        self.track_data("Loss / Entropy loss", entropy_loss.item())
                        self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-estimation / bias", self.bias.item())

            #  Ensemble uncertainty metrics (training batch).
            self.track_data("Uncertainty / Q std (mean)",    q_std_train.mean().item())
            self.track_data("Uncertainty / Q std (max)",     q_std_train.max().item())
            self.track_data("Uncertainty / Q std (min)",     q_std_train.min().item())
            self.track_data("Uncertainty / Q mean (mean)",   q_mean_train.mean().item())
            #  Per-critic mean Q (useful for diagnosing ensemble spread).
            for i, q_i in enumerate(q_values_list):
                self.track_data(f"Q-network / Q{i+1} ensemble mean", q_i.mean().item())
 
            if self._q_method == "iql":
                self.track_data("Value / V (max)",  torch.max(v_values).item())
                self.track_data("Value / V (min)",  torch.min(v_values).item())
                self.track_data("Value / V (mean)", torch.mean(v_values).item())
 
                self.track_data("Target / Q backup (max)",  torch.max(q_backup).item())
                self.track_data("Target / Q backup (min)",  torch.min(q_backup).item())
                self.track_data("Target / Q backup (mean)", torch.mean(q_backup).item())

            else:
                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            self.track_data("Target / sampled_rewards (mean)", torch.mean(sampled_exp_rewards).item())


            if self.policy_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
            if self.critic_scheduler:
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
