# agent.py
import argparse
import imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from env import QuadrupedEnv
from callbacks import StepLoggingCallback

def train(model_path: str, total_timesteps: int):
    env = QuadrupedEnv(model_path, render_mode=None)
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        batch_size=256,
        buffer_size=int(1e6),
        learning_rate=3e-3,
        tau=0.01,
        gamma=0.99,
        ent_coef="auto",
    )
    cb = StepLoggingCallback(out_csv="logs/step_log.csv")
    model.learn(total_timesteps=total_timesteps, callback=cb)
    env.close()
    model.save("sac_tesbot")
    return model

def visualize(model, model_path: str, eval_steps: int, render_mode: str):
    env = QuadrupedEnv(model_path, render_mode=render_mode)
    obs, _ = env.reset()
    frames = []

    for step in range(eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        print(
            f"[{render_mode}] Step {step:4d} ▶ "
            f"reward={info['reward']:.3f}, "
            f"forward_vel={info['forward_vel']:.3f}, "
            f"torso_z={info['torso_z']:.3f}"
        )

        if render_mode == "human":
            env.render()
        else:  # rgb_array
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if done:
            obs, _ = env.reset()

    if render_mode == "rgb_array" and frames:
        imageio.mimsave("eval_run.gif", frames, fps=env.metadata["render_fps"])
        print("Saved ▶ eval_run.gif")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & visualize SAC on Quadruped")
    parser.add_argument("--model-path",     type=str, default="quad_ex.xml")
    parser.add_argument("--total-timesteps",type=int, default=50_000)
    parser.add_argument("--eval-steps",     type=int, default=1_000)
    parser.add_argument("--eval-only",      action="store_true")
    parser.add_argument(
        "--render",
        nargs="?",
        const="rgb_array",
        default="rgb_array",
        choices=["human", "rgb_array"],
        help="Render mode (default: rgb_array; use `--render human` for on-screen)"
    )
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        model = SAC.load("sac_tesbot")
        visualize(model, args.model_path, args.eval_steps, render_mode=args.render)
    elif args.training:
        model = train(args.model_path, args.total_timesteps)
    elif args.render:
        model = train(args.model_path, args.total_timesteps)
        visualize(model, args.model_path, args.eval_steps, render_mode=args.render)
