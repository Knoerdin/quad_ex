import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class StepLoggingCallback(BaseCallback):
    def __init__(self, out_csv="logs/step_log.csv", verbose=0):
        super().__init__(verbose)
        self.out_csv = out_csv
        self.logs = []

    def _on_step(self) -> bool:
        # Try to grab per-step infos/rewards if available
        infos = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)

        if infos is not None and rewards is not None:
            # normal training step
            info   = infos[0]
            reward = float(rewards[0])

            self.logs.append({
                "timestep":     self.num_timesteps,
                "reward":       reward,
                "forward_vel":  info.get("forward_vel",  np.nan),
                "torso_z":      info.get("torso_z",      np.nan),
                "ctrl_cost":    info.get("ctrl_cost",    np.nan),
                "alive_bonus":  info.get("alive_bonus",  np.nan),
                "fall_penalty": info.get("fall_penalty", np.nan),
            })
        else:
            # fallback inside evaluate_policy (no 'infos', 'rewards')
            ep_rews = self.locals.get("episode_rewards", None)
            if ep_rews:
                # log the last completed episode return
                reward = float(ep_rews[-1])
                self.logs.append({
                    "timestep":     self.num_timesteps,
                    "reward":       reward,
                    "forward_vel":  np.nan,
                    "torso_z":      np.nan,
                    "ctrl_cost":    np.nan,
                    "alive_bonus":  np.nan,
                    "fall_penalty": np.nan,
                })
            # if there's nothing to log, just continue
        return True

    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        df = pd.DataFrame(self.logs)
        df.to_csv(self.out_csv, index=False)
        print(f"Saved ▶ {self.out_csv}")
