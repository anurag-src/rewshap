#from teleop import collect_demos
from rew import *
from pyglet.window import key
import keyboard
from pynput import keyboard
import time

def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    '''Wrap Environment here'''
    env = PotentialBasedWrapperTest(gym.make("MountainCar-v0",render_mode='single_rgb_array'))
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos

action = 1

def on_press(key):
    global action
    try:
        if key == keyboard.Key.right:
            action = 2
        elif key == keyboard.Key.left:
            action = 0
    except AttributeError:
        pass

def on_release(key):
    global action
    if (
        key == keyboard.Key.up
        or key == keyboard.Key.down
        or key == keyboard.Key.left
        or key == keyboard.Key.right
        or key == keyboard.Key.space
    ):
        action = 1

def get_action():
    return action
def scale(x):
    return ((x*1.8)/600) - 1.2

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

'''Wrap Environment here'''
env = PotentialBasedWrapperTest(gym.make('MountainCar-v0'))

env.reset()
pygame.init()
done = False
env.render(mode="human")
observation = env.reset()
x = 0
total = 0
next_obs = env.state
while not done:
    env.render(mode="human")
    left, middle, right = pygame.mouse.get_pressed()
    if left:
        x, y = pygame.mouse.get_pos()
        x = scale(x)
        #print(x)
    #print("Position :",next_obs[0])
    next_obs, reward, done, info = env.step(action)
    time.sleep(0.1)
    total = total + reward
print(total)
