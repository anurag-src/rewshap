from rew import *
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
import os
import time
# Saving logs to visulise in Tensorboard, saving models

# The learning agent and hyperparameters
models_dir = "/Users/anuragaribandi/Documents/IndStudy/RewardShaping/models/PPOPotentialTest-0"
logdir = "/Users/anuragaribandi/Documents/IndStudy/RewardShaping/logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
#env = ExtremeWrapperL(gym.make('MountainCar-v0'))
#env = ExtremeWrapperR(gym.make('MountainCar-v0'))
env = PotentialBasedWrapperTest(gym.make('MountainCar-v0'))
env.reset()
model = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=256,
    ent_coef=0,
    learning_rate=7.77e-05,
    n_epochs=4,
    n_steps=16,
    gae_lambda=0.98,
    gamma=0.99,
    verbose=1,
    tensorboard_log=logdir
    )
TIMESTEPS = 1000
for i in range(1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPOPotentialTest-0")
    model.save(f"{models_dir}/{TIMESTEPS * i}")