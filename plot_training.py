#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the Monitor CSV (must be in the same directory)
    df = pd.read_csv('monitor.csv', comment='#')

    # Episode indices
    episodes = df.index + 1

    # 1) Episode Reward over Episodes
    plt.figure()
    plt.plot(episodes, df['r'])
    plt.title('Episode Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

    # 2) Episode Length over Episodes
    plt.figure()
    plt.plot(episodes, df['l'])
    plt.title('Episode Length over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.grid(True)
    plt.show()

    # 3) Reward vs Training Time
    plt.figure()
    plt.plot(df['t'], df['r'])
    plt.title('Reward over Training Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
