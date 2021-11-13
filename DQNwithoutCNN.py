import argparse

import crafter
import stable_baselines3
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-dqn/0')
parser.add_argument('--steps', type=float, default=10000)
args = parser.parse_args()

env = crafter.Env()
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_episode=False,
    save_video=False,
)

model = stable_baselines3.DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=args.steps)

