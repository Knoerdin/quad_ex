import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# Optional MuJoCo viewer for rendering
try:
    from mujoco.viewer import Viewer
except ImportError:
    Viewer = None

class QuadrupedEnv(gym.Env):
    """
    A MuJoCo-based quadruped environment with:
      - free-floating base joint
      - 8 hinge joints (knees + ankles)
      - dense reward: forward velocity, control cost, alive bonus
      - termination on fall (torso height < 0.1m)
      - optional human / rgb_array rendering
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, model_path: str, render_mode: str = None):
        # Load model & data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.dt    = self.model.opt.timestep

        # Rendering
        self.render_mode = render_mode
        self.viewer     = None

        # Ensure actuators present
        assert self.model.nu == 8, f"Expected 8 actuators, found {self.model.nu}"

        # Define action & observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,), dtype=np.float32
        )
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # For reward calculation
        # Initialize forward-position tracker with base x
        self.prev_x = 0.0

    def reset(self, seed=None, options=None):
        # Standard Gymnasium API: returns obs, info
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        # Track base x-position (body 1)
        self.prev_x = float(self.data.xpos[1][0])
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Apply action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Observation
        obs = self._get_obs()

        # Reward calculation
        x = float(self.data.xpos[1][0])
        forward_vel = (x - self.prev_x) / self.dt
        self.prev_x = x
        ctrl_cost = 1e-3 * np.sum(np.square(action))
        reward = forward_vel - ctrl_cost + 0.1

        # Termination on fall
        z = float(self.data.xpos[1][2])
        terminated = z < 0.1
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.ravel(),
            self.data.qvel.ravel()
        ]).astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            if Viewer is None:
                raise ImportError("MuJoCo viewer not available.")
            if self.viewer is None:
                self.viewer = Viewer(self.model, self.data)
            self.viewer.render()
        elif self.render_mode == "rgb_array":
            width, height = 640, 480
            return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None