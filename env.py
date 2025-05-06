# env.py
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import time

class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str,
        render_mode: str = None,
        max_steps: int = 1000,
        min_height: float = 0.12,
    ):
        # --- Load ---
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)

        # --- Spaces ---
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,), dtype=np.float32
        )
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float64
        )

        # --- Params ---
        self.max_steps    = max_steps
        self.min_height   = min_height
        self.elapsed_steps = 0
        self.last_x_pos   = 0.0

        # --- Rendering ---
        assert render_mode in (None, "human", "rgb_array"), f"bad render_mode {render_mode}"
        self.render_mode = render_mode
        self.viewer      = None
        self.renderer    = None

        if render_mode == "rgb_array":
            # off-screen only needs the model
            self.renderer = mujoco.Renderer(self.model)

    def step(self, action):
        # Clip and apply action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        # advance sim
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps += 1

        # Compute observation
        obs = np.concatenate([self.data.qpos, self.data.qvel])

        # Termination: check height
        torso_z   = float(self.data.qpos[2])
        terminated = (torso_z < self.min_height)
        # truncated  = (self.elapsed_steps >= self.max_steps)

        # —— SHAPED REWARD ——
        x_pos = float(self.data.qpos[0])
        forward_vel = (x_pos - self.last_x_pos) # Normalize by time step
        self.last_x_pos = x_pos

        # Basic components
        alive_bonus = 1.0 if not terminated else 0.0  # Normalize
        ctrl_cost = 1e-2 * np.square(action).sum()
        fall_penalty = -10.0 if terminated else 0.0

        # Add orientation and uprightness terms
        up_vector = self.data.geom_xmat[0, 2]  # Assumes z-up, check if this fits your model
        upright_bonus = up_vector  # Between -1 and 1, max when upright

        reward = (
            2.0 * forward_vel        # Encourage walking forward
            - 0.1 * ctrl_cost        # Penalize high-effort actions
            + 1 * upright_bonus    # Bonus for staying upright
            + 1.0 * alive_bonus      # Constant reward for not dying
            + fall_penalty           # Harsh penalty for falling
        )


        info = {
            "torso_z": torso_z,
            "forward_vel": forward_vel,
            "ctrl_cost": ctrl_cost,
            "alive_bonus": alive_bonus,
            "fall_penalty": fall_penalty,
            "reward": reward,
            "terminated": terminated,
            "truncated": False,
        }
        return obs, reward, terminated, False, info

    def reset(self, **kwargs):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps = 0
        self.last_x_pos   = float(self.data.qpos[0])
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
            # first call to render → open the window now
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            time.sleep(1.0 / self.metadata["render_fps"])
            
        elif self.render_mode == "rgb_array":
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None
