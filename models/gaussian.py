from __future__ import annotations

from typing import Any, Literal

import torch
from torch.distributions import Normal
import torch.nn.functional as F
import math

from skrl.utils.spaces.torch import compute_space_limits


# speed up distribution construction by disabling checking
Normal.set_default_validate_args(False)


class GaussianMixin:
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        clip_mean_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: Literal["mean", "sum", "prod", "none"] = "sum",
        role: str = "",
    ) -> None:
        """Gaussian mixin model (stochastic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param clip_mean_actions: Flag to indicate whether the mean actions should be clipped to the action space.
            If ``True``, the mean actions will be clipped before sampling the actions.
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
        :param reduction: Reduction method for returning the log probability density function.
            If ``"none"``, the log probability density function is returned as a tensor of shape
            ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
        :param role: Role played by the model.

        :raises ValueError: If the reduction method is not valid.
        """
        self._g_clip_actions = clip_actions
        self._g_clip_mean_actions = clip_mean_actions
        self._g_min_actions, self._g_max_actions = (
            torch.tensor(-1.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )


        self._g_clip_log_std = clip_log_std
        self._g_min_log_std = min_log_std
        self._g_max_log_std = max_log_std

        self._g_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """Act stochastically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing the following extra output values:

            - ``"log_std"``: log of the standard deviation.
            - ``"log_prob"``: log of the probability density function.
            - ``"mean_actions"``: mean actions (network output after optional clipping).
        """
        # map from observations/states to mean actions and log standard deviations
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        # clip log standard deviations
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        # distribution
        self._g_distribution = Normal(mean_actions, log_std.exp())

        # sample using the reparameterization trick
        x = self._g_distribution.rsample()          # x ~ N(mu, std)
        y = torch.tanh(x)                         # squash to (-1, 1)

        # log of the probability density function
        log_prob = self._g_distribution.log_prob(x)
        log_prob -= torch.log(1 - y.pow(2) + 1e-6)  # Jacobian correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        actions     = y
        mean_action = torch.tanh(mean_actions)              # deterministic action (eval)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_action
        return actions, outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model.

        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)

    def distribution(self, *, role: str = "") -> torch.distributions.Normal:
        """Get the current distribution of the model.

        :param role: Role played by the model.

        :return: Distribution of the model.
        """
        return self._g_distribution
        
    def evaluate(self, inputs: dict, taken_actions: torch.Tensor) -> torch.Tensor:
        mean_actions, outputs = self.compute(inputs, role="policy")
        log_std = outputs["log_std"]

        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_min_log_std, self._g_max_log_std)
            outputs["log_std"] = log_std

        std = log_std.exp()
        dist = Normal(mean_actions, std)
        self._g_distribution = dist

        # clamp BEFORE atanh
        a = taken_actions.clamp(-0.999999, 0.999999)

        # inverse tanh
        x = torch.atanh(a)

        # Gaussian log-prob
        log_prob = dist.log_prob(x)

        # stable Jacobian correction (recommended)
        log_prob -= 2 * (math.log(2) - x - F.softplus(-2 * x))

        return log_prob.sum(dim=-1, keepdim=True)
