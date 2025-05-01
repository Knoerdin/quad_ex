import numpy as np
import gym
from gym import spaces
import mujoco

# Try to import the MuJoCo Python viewer; may not be present in older mujoco versions
try:
    from mujoco.viewer import Viewer
except ImportError:
    Viewer = None

class QuadrupedEnv(gym.Env):
    """
    A MuJoCo-based quadruped environment with:
      - dense reward (forward velocity, control cost, alive bonus)
      - early termination on fall
      - optional rendering via `render_mode`
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        # NOTE: you could set render_fps dynamically to 1/self.dt if you like
        "render_fps": 60  
    }

    def __init__(self, model_path: str, render_mode: str = None):
        # — MuJoCo model & data —
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.dt    = self.model.opt.timestep

        # — Action & observation spaces —
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,), dtype=np.float32
        )
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # — Reward tracking —
        self.prev_x = 0.0

        # — Rendering —
        self.render_mode = render_mode
        self.viewer     = None

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.prev_x = float(self.data.qpos[0])
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs    = self._get_obs()
        reward = self._compute_reward(action)
        done   = self._compute_done()
        info   = {}
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        x = float(self.data.qpos[0])
        forward_vel = (x - self.prev_x) / self.dt
        self.prev_x = x

        ctrl_cost   = 1e-3 * np.sum(np.square(action))
        alive_bonus = 0.1
        return forward_vel - ctrl_cost + alive_bonus

    def _compute_done(self) -> bool:
        z = float(self.data.qpos[2])
        return z < 0.20

    def render(self, mode: str = None):
        """
        mode: "human" or "rgb_array"
        """
        mode = mode or self.render_mode
        if mode == "human":
            if Viewer is None:
                raise ImportError(
                    "Cannot render in human mode: mujoco.viewer not available."
                )
            if self.viewer is None:
                # Create a single Viewer instance
                self.viewer = Viewer(self.model, self.data)
            # This will open a window and draw the current frame
            self.viewer.render()
        elif mode == "rgb_array":
            # If you need an offscreen buffer, you'd set up an MjrContext
            # and call mujoco.mjr_render into it, then read pixels.
            # For now we return a dummy array placeholder:
            width, height = 640, 480
            return np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # no-op if render_mode is None
            return

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None