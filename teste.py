import rsoccer_gym.vss.env_vss.vss_gym as vss_gym
import numpy as np

env = vss_gym.VSSEnv(render_mode="human")

env.reset()

state_space_size = env.observation_space.shape.count    
action_space_size = env.action_space.shape.count

#Creating a q-table and intialising all values as 0
q_table = np.zeros((state_space_size,action_space_size))
print(q_table)
