import gym
import d4rl
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
from utils import save, collect_random
import random
from agent import IQL
from torch.utils.data import DataLoader, TensorDataset
import pickle
import pandas as pd

''' Global variables for defining experience tuple'''
LAST_RATE = 0
STATE = 1
ACTION1 = 2
ACTION2 = 3
NEXT_STATE = 4
CUR_RATE = 5
REWARD = 6

def prep_dataloader():
    path = "adr_data/monotonic/0.pkl"
    with open(path,'rb') as f:
        buffer = pickle.load(f) # buffer is an python list of 332 episodes, where each episodes have step size 3 ==> 332*3 = 996 experience arrays in total. ==> terminal episodes are i*3-1
                                # each experience is len==7, where experience = ['last_rate', 'state', 'action1', 'action2', 'next_state', 'cur_rate', 'reward']
    return buffer

def train(n_observations = 8, 
          action_dims = [16,17], #core/uncore frequency 
          lr = 1e-4, 
          hidden_size = 32, 
          temperature = 3,
          expectile = 0.7,
          tau = 0.005,
          device = 'cpu'):

    buffer = prep_dataloader()
    average10 = deque(maxlen=10)
    
    with wandb.init(project="rlsys-IQL"):
        
        agent = IQL(state_size = n_observations,
                    action_dims = action_dims,
                    learning_rate = lr,
                    tau = tau,
                    temperature = temperature,
                    expectile = expectile,
                    device = device)

        # wandb.watch(agent, log="gradients", log_freq=10)
        # wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches}, step=batches)
        for i in range(len(buffer)):
            experience = buffer[i]
            last_rate, state, action1, action2, next_state, curr_rate, reward = experience
            last_rate = last_rate.to(device)
            state = state.to(device)
            action1 = action1.to(device)
            action2 = action2.to(device)
            next_state = next_state.to(device)
            curr_rate = curr_rate.to(device)
            reward = reward.to(device)
            if i%3 == 0:
                done = True
            else:
                done = False
            policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn(( last_rate, state, action1, action2, next_state, curr_rate, reward, done))

            print("Episode: {} | Reward: {} | Polciy Loss: {}".format(i, reward, policy_loss))
            

if __name__ == "__main__":
    train()
