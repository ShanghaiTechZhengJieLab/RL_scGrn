"""
Policy Gradient, Reinforcement Learning.

Grn inference
"""

import grnenv
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
import random
import numpy as np
#DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = grnenv.GrnEnv(bt_data,bt_bool,bt_dis,bt_energy)
env.seed(1)     # reproducible, general Policy gradient has high variance


def get_final_network(action_set):
    ## network is a matrix
    total = sum(action_set)
    mean = np.mean(total)
    var = np.var(total)
    line = mean+0.5*var
    network = np.array()
    for x in total:
        if(x>line):
            network.append(1)
        else:
            network.append(0)
    network = network.reshape(env.numGene,env.numGene)
    return network


RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.numGene,#observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)
action_set = []#np.array()
rank = 280
for i_episode in range(300):

    observation = env.reset()
    idx = 0
    while True:quit
        if RENDER: env.render()        
        action = RL.choose_action(observation)
        if(i_episode>rank):
            action_set.append(action)
        observation_, reward, done, info = env.step(action)
        if(idx==env.numCell*1):
            done = True
        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
        idx+=1
        observation = observation_
network = get_final_network(action_set)
print(network)