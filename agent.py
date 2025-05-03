# agent.py

import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from env import QuadrupedEnv
import imageio
import time


def train(model_path: str, total_timesteps: int, render_freq: int):
    # 1) base env with human render
    base_env = QuadrupedEnv(model_path, render_mode=None)

    train_env = Monitor(base_env, filename="monitor.csv")
    print("Training environment created")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        batch_size=256,
        buffer_size=int(1e6),
        learning_rate=1e-4,
        tau=0.1,
        gamma=0.99,
        ent_coef="auto",
        device="cuda"
    )

    # 4) learn
    print("starting Training …")
    model.learn(total_timesteps=total_timesteps)
    print("Training complete")
    obs, info = train_env.reset()
    model.save("sac_tesbot")
    train_env.close()
    return model


def visualize(model, model_path: str, eval_steps: int, render_mode: str = "rgb_array"):
    print("Visualizing …")
    eval_env = QuadrupedEnv(model_path, render_mode=render_mode)
    obs, _ = eval_env.reset()
    frames = []  # Moved outside loop

    for step in range(eval_steps):
        print(f"\n[Eval] Step #{step}")
        
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Add exploration noise
        noise = np.random.normal(0, 0.1, size=action.shape)
        action += noise
        print(f"Action (with noise): {action}")
        print(f"Action norm: {np.linalg.norm(action):.4f}")
        
        # Step environment
        obs, reward, terminated, truncated, info = eval_env.step(action)

        # Debug info
        print(f"Reward: {reward:.4f}")
        print(f"Torso Z: {info.get('torso_z', 'N/A'):.4f}")
        print(f"Forward velocity: {info.get('forward_vel', 'N/A'):.4f}")
        print(f"Control cost: {info.get('ctrl_cost', 'N/A'):.4f}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")

        # Render and store frame
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            print("Episode ended — resetting environment.")
            obs, _ = eval_env.reset()

    # Save video/gif
    if frames:
        print("Saving GIF to eval_run.gif …")
        imageio.mimsave("eval_run.gif", frames, fps=30)
    else:
        print("No frames captured — check render_mode")

    print("Visualization complete")
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
        print("Model loaded")
        visualize(
            model,
            model_path=args.model_path,
            eval_steps=args.eval_steps,
            render_mode=args.render
        )
    elif args.training:
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
