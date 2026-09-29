from __future__ import annotations

import dataclasses
import sys
import tqdm
import torch

from skrl.trainers.torch import Trainer, TrainerCfg
from skrl.agents.torch import Agent


@dataclasses.dataclass(kw_only=True)
class OfflineTrainerCfg(TrainerCfg):
    """Configuration for the offline (no-env) trainer."""
    # TrainerCfg already provides: timesteps, headless, disable_progressbar,
    # checkpoint_interval, write_interval — all honoured here.


class OfflineTrainer:
    """Minimal trainer for BC / offline RL — no environment required.

    Replicates the SequentialTrainer's bookkeeping (experiment init,
    write-interval flushing, checkpoint saving) without any env interaction.

    Usage::

        trainer = OfflineTrainer(agent=agent, cfg={"timesteps": 5000})
        trainer.train()
    """

    def __init__(
        self,
        *,
        agent: Agent,
        cfg: OfflineTrainerCfg | dict = {},
    ) -> None:
        self.cfg = OfflineTrainerCfg(**cfg) if isinstance(cfg, dict) else cfg
        self.agent = agent

        # Mirror what SequentialTrainer.__init__ does:
        # initialise the agent and pass trainer_cfg so that "auto"
        # write / checkpoint intervals resolve against cfg.timesteps.
        self.agent.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """Run the offline training loop.

        Each timestep:
          1. ``agent.pre_interaction``  (no-op in BC, but keeps the contract)
          2. ``agent.post_interaction`` → calls ``update()`` then flushes
             tracked data to TensorBoard via ``super().post_interaction()``
        """
        self.agent.enable_models_training_mode(True)

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
        ):
            self.agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)
            self.agent.post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

        # Final checkpoint flush (mirrors Trainer.train behaviour)
        self.agent.write_checkpoint(timestep=self.cfg.timesteps - 1, timesteps=self.cfg.timesteps)

    def eval(self) -> None:
        """Run an offline evaluation loop (no gradient updates)."""
        self.agent.enable_models_training_mode(False)

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
        ):
            self.agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)
            # Call base Agent.post_interaction directly to flush tracking
            # data WITHOUT triggering update() (same pattern as SequentialTrainer.eval)
            super(self.agent.__class__, self.agent).post_interaction(
                timestep=timestep, timesteps=self.cfg.timesteps
            )