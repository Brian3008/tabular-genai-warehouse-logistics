import gymnasium as gym
import rware

env = gym.make("rware-tiny-2ag-v2")
obs, info = env.reset()
print("RWARE working:", obs)