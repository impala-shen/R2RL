# Replay and Reinforce RL (R²RL)

Code for R²RL and considered baselines

## Environment setup

The existing Conda environment uses **Python 3.10.19**, PyTorch **2.10.0**, skrl **2.0.0**, Gymnasium **1.3.0**, and UR RTDE **1.6.2**.

## Components and roles

| Component | Role |
| --- | --- |
| [task_R2RL.py](task_R2RL.py) | Connect the robot, build networks/replay memory, and run training or evaluation |
| [task_settings.py](task_settings.py) | Select the task, environment, home pose, episode limit, and demonstration file |
| [algorithms/OneShot/](algorithms/OneShot/) | Main R²RL algorithm |
| [envs/](envs/) | Interface to the skrl trainer |
| [models/gaussian.py](models/gaussian.py) | Stochastic policy implementation |
| [robotiq/](robotiq/) | Gripper control through RTDE |
| [data/demonstrations/](data/demonstrations/) | Recorded robot transitions used for learning |
| skrl `SequentialTrainer` | Coordinate environment interaction and agent updates |
| UR RTDE | Send commands and read measurements; the UR controller executes motion |

Other entry points are `task_IBRL.py` (imitation-bootstrapped RL), `task_ResidualRL.py` (base controller plus learned corrections), and `task_BC_noenv.py` (unfinished offline BC). `OfflineTrainer.py` runs updates without robot rollouts.


```bash
conda activate offline_rl
```

To create a separate environment using the recorded package baseline:

```bash
conda create -n r2rl python=3.10.19
conda activate r2rl
python -m pip install torch==2.10.0 gymnasium==1.3.0 ur_rtde==1.6.2 numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 tensorboard==2.20.0 tqdm==4.67.3 packaging==25.0
python -m pip install "git+https://github.com/Toni-SM/skrl.git@5a078cff1a611d51eb2164101cdd6fa84e3eec2d"
```

The existing environment was checked; the fresh installation above has not been tested. The original package snapshot is [docs/environment-original.txt](docs/environment-original.txt). CUDA is used when available; otherwise the scripts use CPU.

## Task configuration

Run commands from the repository root with network access to the robot. Configure the task's home pose and environment reset waypoints for your fixture before running.

| Option | Purpose |
| --- | --- |
| `--task` | Choose `snapfit`, `peginhole`, `assembly`, or `openbox`; default is `snapfit` |
| `--robot-ip` | Robot address; default `192.168.1.11` |
| `--home-pose X Y Z RX RY RZ` | Override home pose: meters and rotation-vector radians |
| `--max-steps` | Episode limit; default 100 |
| `--demo-file` | Override the selected task's demonstration CSV |

Edit `TASKS` in `task_settings.py` to change task presets, or `DEFAULT_TASK` in `task_R2RL.py` to change the default selection. Snap-fit, peg-in-hole, and assembly have included demonstrations. **Open-box requires your own `--demo-file`.** Demonstrations must match the task's observation dimensions, normalization, and action frame.

Edit the `cfg` block in `task_R2RL.py` for RL hyperparameters. Defaults include batch size 64, learning rate `1e-4`, discount 0.99, one gradient step per update, five critics, IQL critic updates, and a `DDPG+BC` policy objective. With 100-step episodes, the training budget is 5,000 environment steps.

## Train and evaluate

```bash
# Train from scratch.
python task_R2RL.py --task snapfit --robot-ip 192.168.1.11

# Continue training from a compatible checkpoint.
python task_R2RL.py --task snapfit --checkpoint path/to/agent.pt

# Evaluate the policy mean; a checkpoint is required.
python task_R2RL.py --task snapfit --eval --checkpoint path/to/agent.pt --eval-episodes 30

# View all available options.
python task_R2RL.py --help
```

Add `--stochastic-evaluation` to sample actions during evaluation. Evaluation does not require demonstration data. Checkpoints must match the selected task and network architecture; none are included.

## Logs and checkpoints

Outputs are saved under `runs/torch/<task>/osrl/<experiment>/`, with checkpoints in `checkpoints/`. The `osrl` directory name is retained from the original implementation.

```bash
tensorboard --logdir runs/torch
```
