import stable_baselines3
import gym 
import crafter
import random
import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
import matplotlib.pyplot as plt
import argparse
import pathlib
import time




env=gym.make('CrafterNoReward-v1')

height, width, channels = env.observation_space.shape
env = crafter.Env(seed=100)
actions = env.action_space.n


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(height, width, channels, actions)
print(model.summary())

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=100, window_length=3)
    DQN = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=100
                  )
    return DQN

DQN = build_agent(model, actions)
DQN.compile(Adam(learning_rate=1e-4))

#DQN.fit(env, nb_steps=100, visualize=False, verbose=2)

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-dqnccn/0')
parser.add_argument('--steps', type=float, default=100)
args = parser.parse_args() 


model = stable_baselines3.DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=args.steps)

    
    




