# agent.py

import argparse
from stable_baselines3 import SAC
from env import QuadrupedEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true",
                        help="After training, run a visualization rollout.")
    parser.add_argument("--model-path", type=str, default="quad_ex.xml")
    args = parser.parse_args()

    # 1) Train headless (no rendering)
    train_env = QuadrupedEnv(args.model_path, render_mode=None)
    model = SAC("MlpPolicy", train_env, verbose=1,
                buffer_size=int(1e6), batch_size=256, gamma=0.99, tau=0.005)
    model.learn(total_timesteps=200_000)
    model.save("sac_tesbot")

    # 2) If user asked for --render, launch a new env and visualize
    if args.render:
        eval_env = QuadrupedEnv(args.model_path, render_mode="human")
        obs, _ = eval_env.reset()
        eval_env.render()        # draw initial frame
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = eval_env.step(action)
            eval_env.render()
            if done:
                obs, _ = eval_env.reset()
                eval_env.render()
        eval_env.close()
