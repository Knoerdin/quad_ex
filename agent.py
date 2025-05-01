import argparse
from stable_baselines3 import SAC
from env import QuadrupedEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--model-path",
        type=str,
        default="quad_ex.xml",
        help="Path to your MJCF file with actuators"
    )
    args = parser.parse_args()

    env = QuadrupedEnv(
        model_path=args.model_path,
        render_mode="human" if args.render else None
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        # you can tweak these if you like:
        buffer_size=int(1e6),
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
    )

    # Train (no rendering inside learn—it’ll slow things way down)
    model.learn(total_timesteps=200_000)

    model.save("sac_tesbot")

    # Evaluate with rendering if you want
    if args.render:
        obs = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            env.render()
            if done:
                obs = env.reset()
