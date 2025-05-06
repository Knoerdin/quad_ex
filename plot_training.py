#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(path):
    # Load the Monitor CSV (must be in the same directory)
    df = pd.read_csv(path, comment='#')

    # Episode indices
    episodes = df.index + 1

    # 1) Episode Reward over Episodes
    plt.figure()
    plt.plot(episodes, df['reward'])
    plt.plot(episodes, df['forward_vel'], label='Forward Velocity')
    plt.plot(episodes, df['torso_z'], label='Torso Height')
    plt.plot(episodes, df['ctrl_cost'], label='Control Cost')
    plt.plot(episodes, df['alive_bonus'], label='Alive Bonus')
    plt.plot(episodes, df['fall_penalty'], label='Fall Penalty')
    plt.legend()
    plt.title('Episode Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

    # 2) Episode Length over Episodes
    plt.figure()
    plt.plot(episodes, df['reward'].cumsum())
    plt.title('Episode Length over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.grid(True)
    plt.show()

    # 3) Reward vs Training Time
    plt.figure()
    plt.plot(episodes, df['r'])
    plt.title('Reward over Training Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train & visualize SAC on Quadruped")
    parser.add_argument(
        '--path', type=str, default="step_log.csv",
        help="Path to the step log CSV file"
    )
    args = parser.parse_args()
    main(args.path)
