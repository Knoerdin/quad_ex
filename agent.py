# agent.py

import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from env import QuadrupedEnv

class RenderWrapper(gym.Env):
    """
    Wraps an env to:
      - print on every step()
      - call render() every render_freq steps
    """
    def __init__(self, env: gym.Env, render_freq: int = 1):
        super().__init__()
        self.env = env
        self.render_freq = render_freq
        self.step_count = 0

        # expose the same spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        self.step_count += 1

        # call through to real env.step()
        obs, reward, terminated, truncated, info = self.env.step(action)

        # debug print
        print(f"[Env.step] #{self.step_count}  terminated={terminated}  truncated={truncated}")

        # render every N steps
        if self.step_count % self.render_freq == 0:
            self.env.render()

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        return self.env.close()

def train(model_path: str, total_timesteps: int, render_freq: int):
    # 1) base env with human render
    base_env = QuadrupedEnv(model_path, render_mode=None)
    # 2) wrap for prints+visualization
    # 3) Monitor for stats
    train_env = Monitor(base_env, filename="monitor.csv")
    print("Training environment created")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        batch_size=256,
        buffer_size=int(1e6),
        learning_rate=3e-4,
        tau=0.01,
        gamma=0.99,
        ent_coef="auto",
    )

    # 4) learn
    print("starting Training …")
    model.learn(total_timesteps=total_timesteps)
    print("Training complete")
    obs, info = train_env.reset()
    model.save("sac_tesbot")
    train_env.close()
    return model

def visualize(model, model_path: str, eval_steps: int, render_mode: str):
    print("Visualizing …")
    eval_env = QuadrupedEnv(model_path, render_mode=render_mode)
    obs, _ = eval_env.reset()
    eval_env.render()
    for _ in range(eval_steps):
        print(f"[Eval] step #{_}")
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        eval_env.render()
        if terminated or truncated:
            obs, _ = eval_env.reset()
            eval_env.render()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train & visualize SAC on QuadrupedEnv with live rendering"
    )
    parser.add_argument(
        "--model-path", type=str, default="quad_ex.xml",
        help="Path to MJCF file"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=50_000,
        help="Training steps"
    )
    parser.add_argument(
        "--render-freq", type=int, default=5,
        help="How often (in env steps) to render and print"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load existing model"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=1000,
        help="Number of steps to roll out after training"
    )
    parser.add_argument(
        "--render", nargs="?", const="human", choices=["human", "rgb_array"],
        help="If set, run a final visualization"
    )
    parser.add_argument(
        "--training", action="store_true",
        help="If set, run training"
    )
    args = parser.parse_args()

    if args.eval_only:
        print("Loading existing model sac_tesbot.zip …")
        model = SAC.load("sac_tesbot")
    elif args.training:
        print("Training …")
        model = train(
            model_path=args.model_path,
            total_timesteps=args.total_timesteps,
            render_freq=args.render_freq
        )
    else:
        print("Running visualization …")
        model = train(
            model_path=args.model_path,
            total_timesteps=args.total_timesteps,
            render_freq=args.render_freq
        )
        visualize(
            model,
            model_path=args.model_path,
            eval_steps=args.eval_steps,
            render_mode=args.render
        )
