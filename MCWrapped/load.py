import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
import os
import time
from rew import *

models_dir = "/Users/anuragaribandi/Documents/IndStudy/RewardShaping/models/PPOPotentialTest-0"
#models_dir = "models/PPOExtremeR/0"
#models_dir = "models/PPOWrapped-1"

#env = ExtremeWrapperL(gym.make('MountainCar-v0'))  # continuous: LunarLanderContinuous-v2
#env = ExtremeWrapperR(gym.make('MountainCar-v0'))  # continuous: LunarLanderContinuous-v2
#env = RockingWrapper(gym.make('MountainCar-v0'))  # continuous: LunarLanderContinuous-v2
#env = PotentialBasedWrapper(gym.make('MountainCar-v0'))
env = gym.make('MountainCar-v0')
env.reset()

model_path = f"{models_dir}/999000.zip"
model = PPO.load(model_path, env=env, observation_space=env.observation_space, action_space=env.action_space)

episodes = 1

for ep in range(episodes):
    obs = env.reset()
    done = False
    total = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        time.sleep(0.01)
        total = total + rewards
        #print(rewards)
        env.render()
    print(total)
