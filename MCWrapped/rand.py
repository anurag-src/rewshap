from rew import *
from pyglet.window import key
import keyboard
from pynput import keyboard
import time

env = gym.make('MountainCar-v0')

env.reset()
done = False
env.render(mode="human")
observation = env.reset()
x = 0
total = 0
next_obs = env.state
while not done:
    env.render(mode="human")
    #print("Position :",next_obs[0])
    next_obs, reward, done, info = env.step(env.action_space.sample())
    total = total + reward
print(total)