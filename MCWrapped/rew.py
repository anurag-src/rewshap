import gym
from gym.utils.play import play
import argparse
import pygame
#from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device('cpu')
def scale(x):
    return ((x*1.8)/600) - 1.2

class RewardShapeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.check = [-0.75,-0.036,-1.0,0.5]
        self.visited = [False, False, False, False]
        self.epsilon = 0.01
        self.defaulta = 0
    def step(self, action):
        obs, rew, terminated, info = self.env.step(action)
        checkp = self.check
        check = 0
        for i in range(len(checkp)):
            if not self.visited[i]:
                if abs(obs[0] - checkp[i]) > self.epsilon:
                    check = checkp[i]
                    break
                else:
                    self.visited[i] = True
                    check = checkp[i+1]
                    self.switch(self.defaulta)
                    #rew = rew + 20
                    break

        if check != 0:
            rew = -abs(check - obs[0])
            print(rew)
        #print(rew)
        return obs, rew, terminated, info
    def switch(self, a):
        if a ==0:
            self.defaulta = 2
        if a == 2:
            self.defaulta = 0

class StationaryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.check = -0.5
        self.visited = [False, False, False, False]
        self.epsilon = 0.01
        self.defaulta = 0
    def step(self, action):
        obs, rew, terminated, info = self.env.step(action)
        checkp = self.check
        rew = -abs(checkp - obs[0])
        #print(rew)
        return obs, rew, terminated, info
    def switch(self, a):
        if a ==0:
            self.defaulta = 2
        if a == 2:
            self.defaulta = 0

class RockingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.left = -0.6
        self.right = -0.4
        self.dir = 'left'
        self.visited = [False, False, False, False]
        self.epsilon = 0.01
        self.defaulta = 0
    def step(self, action):
        obs, rew, terminated, info = self.env.step(action)
        dir = self.dir
        if dir == 'right':
            check = self.right
        if dir == 'left':
            check = self.left
        if abs(obs[0] - check) < self.epsilon:
            if dir == 'right':
                self.dir = 'left'
                check = self.left
                self.switch(self.defaulta)
            if dir == 'left':
                self.dir = 'right'
                check = self.right
                self.switch(self.defaulta)
        rew = -abs(check - obs[0])
        return obs, rew, terminated, info
    def switch(self, a):
        if a ==0:
            self.defaulta = 2
        if a == 2:
            self.defaulta = 0
class ExtremeWrapperL(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.check = -1.2
        self.visited = [False, False, False, False]
        self.epsilon = 0.01
        self.defaulta = 0
    def step(self, action):
        #print(self.state)
        obs, rew, terminated, info = self.env.step(action)
        checkp = self.check
        rew = -abs(checkp - obs[0])
        return obs, rew, terminated, info
    def switch(self, a):
        if a ==0:
            self.defaulta = 2
        if a == 2:
            self.defaulta = 0
class ExtremeWrapperR(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.check = 0.6
        self.visited = [False, False, False, False]
        self.epsilon = 0.01
        self.defaulta = 0
    def step(self, action):
        obs, rew, terminated, info = self.env.step(action)
        checkp = self.check
        rew = -abs(checkp - obs[0])
        return obs, rew, terminated, info
    def switch(self, a):
        if a ==0:
            self.defaulta = 2
        if a == 2:
            self.defaulta = 0
class PotentialBasedWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.check = [-0.77,-0.036,-0.8,0.5]
        self.visited = [False, False, False, False]
        self.epsilon = 0.1
        self.defaulta = 0
    def step(self, action):
        current = self.state[0]
        obs, rew, terminated, info = self.env.step(action)
        checkp = self.check
        check = 0
        for i in range(len(checkp)):
            if not self.visited[i]:
                if abs(obs[0] - checkp[i]) > self.epsilon:
                    check = checkp[i]
                    break
                else:
                    self.visited[i] = True
                    if i+1 < len(checkp):
                        check = checkp[i+1]
                    break

        if check != 0:
            prevpotential = abs(check - current)
            currentpotential = abs(check - obs[0])
            rew = -(currentpotential - prevpotential)*10
            print("Current Checkpoint :", check)
            print("Reward :", rew)
        #print(rew)
        return obs, rew, terminated, info
class PotentialBasedWrapperTest(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.check = -0.77
        self.visited = [False, False, False, False]
        self.epsilon = 0.1
        self.defaulta = 0
    def step(self, action):
        current = self.state[0]
        obs, rew, terminated, info = self.env.step(action)
        check = self.check
        if check != 0:
            prevpotential = abs(check - current)
            currentpotential = abs(check - obs[0])
            rew = -(currentpotential - prevpotential)*10
            #print("Current Checkpoint :", check)
            #print("Reward :", rew)
        #print(rew)
        return obs, rew, terminated, info
# env = RewardShapeWrapper(gym.make('MountainCar-v0'))
# env.reset()
# pygame.init()
# done = False
# observation = env.reset()
# x = 0
# env.render(mode="human")
# total = 0
# while not done:
#     env.render(mode="human")
#     #a = env.action_space.sample()
#     a = env.defaulta
#     left, middle, right = pygame.mouse.get_pressed()
#     if left:
#         x, y = pygame.mouse.get_pos()
#         x = scale(x)
#         print(x)
#     next_obs, reward, done, info = env.step(a)
#     total = total + reward
# print(total)
#     #print(next_obs[0],reward)