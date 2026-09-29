from __future__ import annotations

from collections.abc import Callable
import dataclasses

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class ResidualRL_CFG(AgentCfg):
    """Configuration for composable residual reinforcement learning updates."""

    deterministic_policy: bool = True
    """Use a deterministic actor. Set to ``False`` for a stochastic actor."""

    q_method: str = "bellman"
    """Critic target: ``"bellman"`` (TD3/SAC style) or ``"iql"``."""

    policy_method: str = "maxQ"
    """Actor objective: ``"maxQ"`` or replay-regularized ``"DDPG+BC"``."""

    gradient_steps: int = 1
    batch_size: int = 128
    discount_factor: float = 0.99
    polyak: float = 0.005

    learning_rate: float | tuple[float, float] = 3e-4
    learning_rate_scheduler: type | tuple[type | None, type | None] | None = None
    learning_rate_scheduler_kwargs: dict | tuple[dict, dict] = dataclasses.field(default_factory=dict)

    observation_preprocessor: type | None = None
    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    state_preprocessor: type | None = None
    state_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)

    random_timesteps: int = 0
    """Steps for which a random residual is added to the base controller."""

    learning_starts: int = 1_000
    grad_norm_clip: float = 0

    exploration_noise: type | None = None
    exploration_noise_kwargs: dict = dataclasses.field(default_factory=dict)
    exploration_scheduler: Callable[[int, int], float] | None = None

    policy_delay: int = 2
    smooth_regularization_noise: type | None = None
    smooth_regularization_noise_kwargs: dict = dataclasses.field(default_factory=dict)
    smooth_regularization_clip: float = 0.5

    initial_entropy_value: float = 0.01
    """Initial/fixed entropy coefficient for a stochastic actor."""

    learn_entropy: bool = True
    target_entropy: float | None = None
    entropy_learning_rate: float = 3e-4

    expectile: float = 0.7
    """IQL expectile used to fit the value function."""

    ddpg_bc_alpha: float = 0.1
    """Q-loss scale numerator in ``lambda = alpha / mean(abs(Q))``."""

    bc_loss_scale: float = 1.0
    """Multiplier on the replay behavior-cloning regularizer."""

    base_policy_type: str = "checkpoint"
    """Base source: ``"checkpoint"`` for a frozen model or ``"one_demo"`` for demo actions."""

    demo_file: str = ""
    """CSV demonstration path used when ``base_policy_type="one_demo"``."""

    num_envs: int = 1
    """Number of parallel environments (used by the one-demo controller)."""

    residual_action_scale: float = 1.0
    """Fraction of the environment half-range available to the residual policy."""

    base_policy_outputs_normalized: bool = True
    """Map the base policy's ``[-1, 1]`` output to the environment action bounds."""

    preprocess_base_observations: bool = False
    """Use the online observation/state preprocessors for the frozen base policy."""

    rewards_shaper: Callable | None = None
    mixed_precision: bool = False

    def validate(self) -> bool:
        self.q_method = self.q_method.lower()
        if self.q_method not in {"bellman", "iql"}:
            raise ValueError("q_method must be 'bellman' or 'iql'")
        policy_method = self.policy_method.lower().replace("_", "").replace("-", "")
        if policy_method == "maxq":
            self.policy_method = "maxQ"
        elif policy_method in {"ddpg+bc", "ddpgbc"}:
            self.policy_method = "DDPG+BC"
        else:
            raise ValueError("policy_method must be 'maxQ' or 'DDPG+BC'")
        if self.base_policy_type not in {"checkpoint", "one_demo"}:
            raise ValueError("base_policy_type must be 'checkpoint' or 'one_demo'")
        if self.base_policy_type == "one_demo" and not self.demo_file:
            raise ValueError("demo_file is required when base_policy_type='one_demo'")
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if not 0 < self.residual_action_scale:
            raise ValueError("residual_action_scale must be greater than 0")
        if self.policy_delay < 1:
            raise ValueError("policy_delay must be at least 1")
        if self.initial_entropy_value <= 0:
            raise ValueError("initial_entropy_value must be greater than 0")
        if self.entropy_learning_rate <= 0:
            raise ValueError("entropy_learning_rate must be greater than 0")
        if not 0 < self.expectile < 1:
            raise ValueError("expectile must be between 0 and 1")
        if self.ddpg_bc_alpha <= 0:
            raise ValueError("ddpg_bc_alpha must be greater than 0")
        if self.bc_loss_scale < 0:
            raise ValueError("bc_loss_scale must be non-negative")
        return True

    def expand(self) -> None:
        super().expand()
        if not isinstance(self.learning_rate, (tuple, list)):
            self.learning_rate = (self.learning_rate, self.learning_rate)
        if self.learning_rate_scheduler is None:
            self.learning_rate_scheduler = (None, None)
        elif not isinstance(self.learning_rate_scheduler, (tuple, list)):
            self.learning_rate_scheduler = (
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
            )
        if not isinstance(self.learning_rate_scheduler_kwargs, (tuple, list)):
            self.learning_rate_scheduler_kwargs = (
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
            )
