import gymnasium as gym
import numpy as np
import time
import torch
from collections import deque
import pandas as pd

from scipy.spatial.transform import Rotation as R

class SnapFitRobotEnv(gym.Env):
    """
    Online gym environment wrapping the real UR robot via RTDE.
    Observations : 19-dim  (e.g. TCP pose 6 + F/T wrench 6)
    Actions      : 6-dim   (delta-pose in original scale, clipped)
    """

    metadata = {"render_modes": []}
    # # ── Task config ───────────────────────────────────────
    # # ── action bounds found in in the demo ───────────────────────────────────────
    # ACTION_MIN = np.array([-0.00080266, -0.00042364, -0.00046984,
    #                        -0.00098748, -0.00063748, -0.00429801], dtype=np.float32)
    # ACTION_MAX = np.array([ 0.00133239,  0.00044393,  0.0021458,
    #                         0.0015833,   0.00063114,  0.00541633], dtype=np.float32)
    # ── Constrained action bounds (changed to mansual defined velocity limit based on task&safe exploration)) ───────────────────────────────────────
    # ACTION_MIN = np.array([-2.21e-04*0.54, -1.01e-04, -8.00e-05, -2.12e-04, -3.35e-04, -3.26e-04], dtype=np.float32)
    # ACTION_MAX = np.array([2.21e-04*0.54, 0.000231, 0.000796*0.9, 0.000385, 0.000279, 0.001546], dtype=np.float32)
    ACTION_MIN = np.array([-0.001, -0.0005, -0.0003, -0.01, -0.01, -0.01], dtype=np.float32)
    ACTION_MAX = np.array([0.001, 0.0005, 0.0007, 0.01, 0.01, 0.01], dtype=np.float32)

# 0.000479*0.82
    FORCE_LIMIT   = 90.0         # [N] safety stop force magnitude # snapfit  60
    CTRL_HZ       = 20          # [Hz]  control frequency
    MOVE_SPEED    = 0.3        # [m/s] moveL speed
    MOVE_ACCEL    = 0.3        # [m/s²]
    

    def __init__(self, rtde_c, rtde_r, home_pose: np.array, max_steps: int = 100, verbose: bool = False, offline_mode = True, device = 'cpu'):
        super().__init__()
        self.rtde_c   = rtde_c
        self.rtde_r   = rtde_r
        self.home     = home_pose      # 6-DOF [x,y,z,rx,ry,rz]
        self.verbose  = verbose
        self._steps   = 0
        self._steps_total = 0
        self.device = device
        self._offline_mode = offline_mode
        self._last_action = np.zeros(6, dtype=np.float32)  # neutral action
        self._MAX_STEPS  = max_steps      # episode length — tune to your task

        self._episode_ret = 0.0        # cumulative episode return (for logging)
        self._prev_obs    = None
        self._accmu_force_x = 0    # for compute reward
        self.mid_pos = home_pose+np.array([0.003, 0,  0.02, 0.,  -0.0, -0.0]) 
        self.height = 0.02  # # snapfit0: 0.02, snapfit1: 0.02, open box: 0.05  
        self.target_z = self.home[2]+self.height # get from the last z position from replay 
         
        # skrl expects a single (non-batched) env here
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(6, dtype=np.float32),
            high= np.ones(6, dtype=np.float32),
        )
        
        # ── Evaluation ───────────────────────────────────────────────────────
        # Assign before training:  env.eval_policy_fn = agent.act_deterministic
        # Signature: action_norm_np = eval_policy_fn(obs_np_float32)  →  np.ndarray (6,)
        self.eval_policy_fn  = None
        self._last_eval_step = 0          # total-step counter when last eval ran
        self._eval_results: list[dict] = []   # full history of eval runs
 
        # skrl expects a single (non-batched) env here
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(6, dtype=np.float32),
            high= np.ones(6, dtype=np.float32),
        )

        df = pd.read_csv("demos/snapfit0/one_demo.csv")

        # ── Extract observations (12 dims) and actions (6 dims) ───────────────────────
        obs_cols    = [f"observations.{i}" for i in range(19)]
        action_cols = [f"actions.{i}" for i in range(6)]

        self._observations = df[obs_cols].values.astype(np.float32)    # (5000, 12)
        actions      = df[action_cols].values.astype(np.float32) # (5000, 6)
 
    # ── helpers ───────────────────────────────────────────────────────────────
    def _denorm(self, action_norm: np.ndarray) -> np.ndarray:
        """Map [-1,1] → original delta-pose space."""
        return (action_norm + 1) / 2 * (self.ACTION_MAX - self.ACTION_MIN + 1e-8) + self.ACTION_MIN


    def normalize_obs(self, obs: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Map obs from [low, high] → [-1, 1], clip outliers."""
        obs_clipped = np.clip(obs, low, high)
        return 2.0 * (obs_clipped - low) / (high - low) - 1.0

    def _get_obs(self) -> np.ndarray:
        if self._offline_mode:
            # offline setting
            # Load the gooddemo file
            obs = self._observations[self._steps+500*0,:]
        else:
            ## online setting
            curr_Force = self.rtde_r.getActualTCPForce()
            curr_TCP = self.rtde_r.getActualTCPPose()

            curr_Force = np.array(curr_Force)
            curr_TCP = np.array(curr_TCP)

            # ── Normalize TCP relative to initial pose ────────────────────────────────
            init_TCP = self.home
            init_xyz = init_TCP[:3]                 # translation
            init_rot = init_TCP[3:]                 # rotation [rx, ry, rz]
            rel_xyz = curr_TCP[:3] - init_xyz

            R_init = R.from_rotvec(init_rot)
            R_curr = R.from_rotvec(curr_TCP[3:])
            R_rel = R_init.inv() * R_curr

            rel_rot = R_rel.as_rotvec()

            rel_TCP = np.concatenate([rel_xyz, rel_rot])

            # ── Build observation ─────────────────────────────────────────────────────
            obs = np.concatenate([rel_TCP.flatten(), curr_Force.flatten(),[self._accmu_force_x]])  # shape (19,)
            # print("obs", obs)
            new_obs = np.concatenate([obs, self._last_action])  # (19,)
            sampled_observations = torch.tensor(obs, dtype=torch.float32).to(self.device) 

            # ── normalize ─────────────────────────────────────────────────────
            OBS_LOW = np.array([
                -0.10, -0.10, -0.10,          # rel xyz  [m]
                -0.10, -0.10, -0.10,       # rel rotvec [rad]
                -100.0, -100.0, -100.0,       # force  Fx Fy Fz [N]
                -10.0,  -10.0,  -10.0,        # torque Tx Ty Tz [Nm]
                -8000,
                *[-1.0] * len(self._last_action),  # last action (already in [-1, 1])
            ])
            OBS_HIGH = -OBS_LOW  # symmetric bounds
            norm_obs = self.normalize_obs(new_obs, OBS_LOW, OBS_HIGH)

        return norm_obs

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        Define your task reward here.
        Example: penalise distance to a goal TCP pose + snap success bonus.
        """
        # ── unpack ────────────────────────────────────────────────────────────
        curr_Force = self.rtde_r.getActualTCPForce()
        curr_TCP = self.rtde_r.getActualTCPPose()
        curr_Force = np.array(curr_Force)
        force_mag = np.linalg.norm(curr_Force[:3])

        self._accmu_force_x += curr_Force[0]
        

        dist = min(self.height, np.linalg.norm(curr_TCP[2] - self.target_z))
        # dist = np.linalg.norm(curr_TCP[2] - self.target_z)
        print("TCP distance to z target:", dist)

        print("self._accmu_force_x ", self._accmu_force_x)
        # print("force_mag", force_mag)
        print("curr_Force[0]", curr_Force[0])
        # print("curr_Force[1]", curr_Force[1])
        # print("curr_Force[2]", curr_Force[2])

        # # open box reward
        # if self._accmu_force_x > 100 and self._accmu_force_x < 10000 and curr_Force[0] > 2: # snapfit0
        #     reward = 10 * (1 - dist/self.height)  #  distance bonus
        #     reward = 0.5 + (1 - dist/self.height)  #  distance bonus
        # # elif self._accmu_force_x > 12000 or force_mag > self.FORCE_LIMIT or curr_Force[0] > 60 or curr_Force[0] < -20 or bool(abs(curr_Force[1]) > 40 or abs(curr_Force[2]) > 50):  #snapfit 0
        # #     reward = 0.  # safety violation
        # else:
        #     reward = 0.


        # snapfit reward
        # if self._accmu_force_x > 300 and self._accmu_force_x < 2000 and curr_Force[0] > 10 and dist < self.height-0.005: # snapfit2
        if self._accmu_force_x > 100 and self._accmu_force_x < 2000 and curr_Force[0] > 5: # snapfit 1 # snap2: 500, 2000  snap1: 80,2000
            reward = 0.3 + 0.7 * (1 - dist/self.height)  #  distance bonus
        # elif self._accmu_force_x > 12000 or force_mag > self.FORCE_LIMIT or curr_Force[0] > 50 or curr_Force[0] < -20 or bool(abs(curr_Force[1]) > 40 or abs(curr_Force[2]) > 40):  #snapfit 0
        # # elif self._accmu_force_x > 12000 or force_mag > self.FORCE_LIMIT or curr_Force[0] > 40 or curr_Force[0] < -10 or bool(abs(curr_Force[1]) > 40 or abs(curr_Force[2]) > 40):  #snapfit 1
        #     reward = 0.  # safety violation
        else:
            reward = 0.
        
        return float(reward)

    def _is_terminated(self, obs: np.ndarray) -> bool:
        """True on safety violation or snap success."""
        curr_Force = self.rtde_r.getActualTCPForce()
        curr_Force = np.array(curr_Force)
        force_mag = float(np.linalg.norm(curr_Force[:3]))
        # print("force_mag", force_mag)
        force_limit_reach = bool(abs(curr_Force[0]) > 50 or abs(curr_Force[1]) > 30 or abs(curr_Force[2]) > 50 or curr_Force[0] < -20)
        timeout = bool(self._steps > self._MAX_STEPS)
        print("force_limit_reach", force_limit_reach)
        e_stop    = force_mag > self.FORCE_LIMIT
        return timeout or e_stop or force_limit_reach
    
    # def _admittance_ctr(self, obs: np.ndarray, action: np.ndarray) -> float:
    #     f

    # ── gym API ───────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        """        Called by the base Trainer at episode start.
        Returns: (obs np.float32 (19,),  info dict)
        """
        super().reset(seed=seed)

        if self.verbose and self._steps > 0:
            print(f"[reset] episode return={self._episode_ret:.4f}  "
                  f"steps={self._steps}")

        curr_TCP = self.rtde_r.getActualTCPPose()
        curr_TCP_mid = np.array(curr_TCP)+np.array([0.02, 0,  0.02, 0.,  -0.0, -0.0]) 
        reset_TCP = np.array([-0.42863826,  -0.1059438,  0.22932025,  2.8398332,   0.60378456, -0.19162114])
        reset_TCP_mid = np.array([-0.42863826,  -0.1059438,  0.22932025,  2.8398332,   0.60378456, -0.19162114])+np.array([0.00, 0,  0.02, 0.,  -0.0, -0.0]) 
        curr_Force = self.rtde_r.getActualTCPForce()
        curr_Force = np.array(curr_Force)
        if self._accmu_force_x > 200 or abs(curr_Force[0]) > 30:
            self.rtde_c.moveL(curr_TCP_mid.tolist(), speed=0.1, acceleration=0.05)
            time.sleep(0.1)
            self.rtde_c.moveL(reset_TCP_mid.tolist(), speed=0.1, acceleration=0.05)
            time.sleep(0.1)
            self.rtde_c.moveL(reset_TCP.tolist(), speed=0.1, acceleration=0.05)

        self._steps       = 0
        self._episode_ret = 0.0
        self._accmu_force_x = 0
        self._last_action = np.zeros(6, dtype=np.float32)

        # ── safe return to home ───────────────────────────────────────────────
        # self.rtde_c.servoStop()
        self.rtde_c.speedStop()
        time.sleep(0.1)
        curr_TCP = self.rtde_r.getActualTCPPose()
        curr_TCP = np.array(curr_TCP)
        if np.linalg.norm(curr_TCP - self.home) > 0.01:
            self.rtde_c.moveL(self.mid_pos.tolist(), speed=0.1, acceleration=0.05)
        self.rtde_c.moveL(self.home.tolist(), speed=0.1, acceleration=0.1)

        time.sleep(0.1)                        # wait for motion to settle
        self.rtde_c.zeroFtSensor()             # zero F/T after reaching home
        time.sleep(0.1)

        obs = self._get_obs()
        self._prev_obs = obs.copy()
        return obs, {}


    # ── Evaluation ───────────────────────────────────────────────────────────
    def _go_home(self) -> None:
        """Move robot back to home pose and re-zero the F/T sensor."""
        # self.rtde_c.servoStop()
        print("before")
        self.rtde_c.speedStop()
        time.sleep(0.3)
        self.rtde_c.moveL(self.home.tolist(), speed=self.MOVE_SPEED, acceleration=self.MOVE_ACCEL)
        print("after")
        time.sleep(0.3)
        self.rtde_c.zeroFtSensor()
        time.sleep(0.1)

    def command_in_tcp_frame(self, mouse_input):

        # Get current TCP pose
        tcp_pose = self.rtde_r.getActualTCPPose()
        current_forces = self.rtde_r.getActualTCPForce()

        # scale_trans
        st = 0.01
        # scale_rot
        sr =0.025

        # command = [-1*mouse_input[1]*st, mouse_input[0]*st, mouse_input[2]*st, -1*mouse_input[3]*sr, mouse_input[4]*sr, mouse_input[5]*sr]
        command = [-1*mouse_input[1]*st, -1*mouse_input[0]*st, -1*mouse_input[2]*st, -1*mouse_input[3]*sr, mouse_input[4]*sr, mouse_input[5]*sr]


        # Frame for base->tcp
        T_tcp_in_base = np.eye(4)
        T_tcp_in_base[:3, :3] = R.from_rotvec(tcp_pose[3:6]).as_matrix()
        T_tcp_in_base[:3, 3] = tcp_pose[:3]

        # # Create target transformation in camera frame
        T_target_R_in_tcp = np.eye(4)
        T_target_R_in_tcp[:3, :3] = R.from_rotvec(command[3:6]).as_matrix()
        T_target_T_in_tcp = np.eye(4)
        T_target_T_in_tcp[:3, 3] = command[:3]

        # # Calculate target in base frame (starting from camera position)
        # T_target_in_base = T_target_T_in_tcp @ T_tcp_in_base @ T_target_R_in_tcp
        T_target_in_base = T_tcp_in_base @ T_target_R_in_tcp @ T_target_T_in_tcp
        # Extract new position and orientation
        new_position = T_target_in_base[:3, 3]
        new_orientation = R.from_matrix(T_target_in_base[:3, :3]).as_rotvec()

        # # Combine and send command
        new_tcp_pose = np.concatenate([new_position, new_orientation])
        return new_tcp_pose

    
    def step(self, action: np.ndarray):
        """
        Called by the base Trainer every timestep.

        action : np.float32 (6,)  in [-1, 1]  — skrl strips the batch dim
                 before handing to the raw gym env via wrap_env

        Returns: obs (19,), reward (float), terminated, truncated, info
        """
        self._steps += 1
        self._steps_total += 1
        # ── sanitise action ───────────────────────────────────────────────────
        action = np.clip(action.astype(np.float32).flatten(), -1.0, 1.0)
        self._last_action = action.copy()   # ← store the normalised action
        TCP_policy  = self._denorm(action)
        # scale_trans
        st = 0.01
        # scale_rot
        sr = 0.025
        # command = [-1*mouse_input[1]*st, mouse_input[0]*st, mouse_input[2]*st, -1*mouse_input[3]*sr, mouse_input[4]*sr, mouse_input[5]*sr]
        # TCP_policy = [-1*TCP_policy[1]*st, -1*TCP_policy[0]*st, -1*TCP_policy[2]*st, -1*TCP_policy[3]*sr, TCP_policy[4]*sr, TCP_policy[5]*sr]

        # ───────────────────────────────────────────────────────────────────────────── Send TCP policy to UR
        # # ─ calculate sent TCP pose
        # # Frame for base->tcp
        # new_tcp_pose = self.command_in_tcp_frame(TCP_policy)  ## for servo


        curr_TCP = np.array(self.rtde_r.getActualTCPPose(), dtype=np.float64)
        T_tcp_in_base = np.eye(4)
        T_tcp_in_base[:3, :3] = R.from_rotvec(curr_TCP[3:6]).as_matrix()  ## tcp_pose is curtcp
        T_tcp_in_base[:3, 3] = curr_TCP[:3]

        # # Create target transformation with policy output
        T_delta = np.eye(4)
        T_delta[:3, :3] = R.from_rotvec(TCP_policy[3:6]).as_matrix()
        T_delta[:3, 3] = TCP_policy[:3]

        T_target_in_base = T_tcp_in_base @ T_delta

        # Extract new position and orientation
        new_position = T_target_in_base[:3, 3]
        new_orientation = R.from_matrix(T_target_in_base[:3, :3]).as_rotvec()

        new_tcp_pose = TCP_policy + curr_TCP
        TCP_policy_vel = TCP_policy/0.02

        # ── apply delta to current TCP pose every 1000 steps ───────────────────────
        should_move = True

        # # if self._offline_mode:
        # should_move = (
        #     self._steps_total > 6000 and
        #     (self._steps_total % 2000) < 250
        # )

        if should_move:
            # self.rtde_c.speedL(
            # TCP_policy_vel,
            # acceleration=self.MOVE_ACCEL,
            # time = 0.02)
            # # time.sleep(0.1)
            
            self.rtde_c.moveL(
                    new_tcp_pose,
                    speed=self.MOVE_SPEED,
                    acceleration=self.MOVE_ACCEL
            )

        #     q_pose = self.rtde_r.getActualQ()
        #     q_target = self.rtde_c.getInverseKinematics(new_tcp_pose)
        #     dist = np.linalg.norm(np.array(q_pose)-np.array(q_target))  
        #     # Print gripper status with the motion info
        #     # print(f"Gripper status: {'OPEN' if gripper_status['is_open'] else 'CLOSED'}")

        #     delay = 0.002
        #     if dist <= 3:  ## the lower the more conservative
        #         # if replay:
        #         #     self.rtde_c.servoJ(q_target, delay, delay, delay, 0.03, 2000)
        #         # else:
        #         #     self.rtde_c.servoJ(q_target, delay, delay, delay, 0.03, 1000)
        #         # self.rtde_c.servoJ(q_target, delay, delay, delay, 0.03, 1000)
        #         # self.rtde_c.servoJ(q_target, delay, delay, delay, 0.03, 1500) # using this one
        #         self.rtde_c.moveL(
        #             new_tcp_pose,
        #             speed=self.MOVE_SPEED,
        #             acceleration=self.MOVE_ACCEL
        #     )
        #         # time.sleep(delay)
        #     else:
        #         print("[Warning] Blocking motion - Dist in command: ", dist)
        # # time.sleep(1.0 / self.CTRL_HZ)

        # ── observe ───────────────────────────────────────────────────────────
        obs        = self._get_obs()
        terminated = self._is_terminated(obs)
        truncated  = (self._steps >= self._MAX_STEPS) and not terminated
        reward     = self._compute_reward(obs, new_tcp_pose)

        self._episode_ret += reward
        self._prev_obs     = obs.copy()

        if self.verbose:
            print(f"  [t={self._steps:03d}] r={reward:+.4f}  "
                  f"Fz={obs[8]:.2f}N")

        # put any per-step scalars here for TensorBoard logging
        info = {
            "episode_return": self._episode_ret,
            "force_magnitude": float(np.linalg.norm(obs[6:9])),
        }

        return obs, float(reward), bool(terminated), bool(truncated), info