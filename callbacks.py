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
        # record whatever you like; here we grab the first env's info & reward
        info   = self.locals["infos"][0]
        reward = float(self.locals["rewards"][0])

        self.logs.append({
            "timestep":    self.num_timesteps,
            "reward":      reward,
            "forward_vel": info.get("forward_vel", np.nan),
            "torso_z":     info.get("torso_z",    np.nan),
            "ctrl_cost":   info.get("ctrl_cost",  np.nan),
            "alive_bonus": info.get("alive_bonus", np.nan),
            "fall_penalty": info.get("fall_penalty", np.nan),
        })
        return True

    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        df = pd.DataFrame(self.logs)
        df.to_csv(self.out_csv, index=False)
        print(f"Saved ▶ {self.out_csv}")