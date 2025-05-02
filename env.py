import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer  # for mujoco.viewer.launch()

class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str,
        render_mode: str = None,
        max_steps: int = 1000,
        min_height: float = 0.003,
        min_upright_cos: float = 0.5
    ):
        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.min_height = min_height
        self.min_upright_cos = min_upright_cos
        self.elapsed_steps = 0

        # Pre-compute body ID for torso
        # Note: mj_name2id expects (model, obj_type:int, name:str)
        self.torso_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            'torso'
        )

        # Define action space: normalized [-1,1] per actuator
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )

        # Define observation space: full qpos+qvel
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64
        )

        # Rendering setup
        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch(self.model, self.data)

    def step(self, action):
        # Clip and apply action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps += 1

        # Compute observation
        obs = np.concatenate([self.data.qpos, self.data.qvel])

        # Termination: check height and orientation
        torso_z = float(self.data.qpos[2])
        # Extract rotation matrix for torso body
        xmat = self.data.xmat[9*self.torso_body_id : 9*(self.torso_body_id+1)]
        upright_cos = xmat[8]
        # local z-axis dot world z-axis = element (2,2) of rotation matrix
        terminated = (torso_z < self.min_height)
        truncated = self.elapsed_steps >= self.max_steps

        # —— SHAPED REWARD ——
        # 1) forward progress (along x)
        x_pos = float(self.data.qpos[0])
        forward_vel = (x_pos - self.last_x_pos) / self.dt
        self.last_x_pos = x_pos

        # 2) small alive bonus for not having fallen over
        alive_bonus = 0.5 if not terminated else 0.0

        # 3) upright posture reward (higher when spine is vertical)
        #    clip at zero so inverted is not “rewarded”
        posture_reward = max(upright_cos, 0.0)

        # 4) control cost (penalize large torques)
        ctrl_cost = 1e-3 * np.sum(np.square(action))

        # 5) optional fall penalty
        fall_penalty = -1.0 if terminated else 0.0

        reward = forward_vel + alive_bonus + 0.1 * posture_reward - ctrl_cost + fall_penalty

        info = {
            "torso_z": torso_z,
            "forward_vel": forward_vel,
            "upright_cos": upright_cos,
            "ctrl_cost": ctrl_cost,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # Gymnasium seeding
        super().reset(seed=seed)
        # Reset simulation data and time
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps = 0
        # Initial observation
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            width, height = 640, 480
            return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
