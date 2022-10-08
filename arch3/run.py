from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
from collections import deque
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import csv
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from ddqn import QLearner, compute_td_loss, ReplayBuffer

MODEL_PATH = './models/pong-'
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 50000000
batch_size = 64
gamma = 0.99
record_idx = 10000

replay_initial = 10000
replay_buffer = ReplayBuffer(50000)
model = QLearner(env, batch_size, gamma, replay_buffer)
#model.load_state_dict(torch.load("model_pretrained.pth", map_location='cpu'))

target_model = QLearner(env, batch_size, gamma, replay_buffer)
target_model.copy_from(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
    print("Using cuda")

epsilon_start = 1.0
epsilon = 0.99
epsilon_final = 0.05
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0
SAVE_MODEL =True
SAVE_MODEL_INTERVAL =10
state = env.reset()

total_step =1
last_100_reward = deque(maxlen=100)

fields = ["Episode","Step","Epsilon","Loss","Reward", "Avg of last 100 Rewards"]
filename = "data1.csv"

#with open(filename,'w') as csvfile:
#    csvwriter = csv.writer(csvfile)
#    csvwriter.writerow(fields)
for episode in range(1, 100000):
    state = env.reset()
    #state = np.stack((state, state, state, state))
    total_reward = 0
    total_loss = 0
    for step in range(100000):

        #epsilon = epsilon_by_frame(frame_idx)
        #print(epsilon)
        action = model.act(state, epsilon)
    
        next_state, reward, done, _ = env.step(action)
        #next_state = np.stack((next_state, state[0], state[1], state[2]))

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward
        
        total_step+=1
        if total_step%500==0:
            if epsilon > epsilon_final:
                epsilon *=0.99


        if len(replay_buffer) > replay_initial:
            loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().numpy())
        else:
            loss = 0
        total_loss += loss
        
        if done:
            
            target_model.copy_from(model)
            if SAVE_MODEL and episode%SAVE_MODEL_INTERVAL==0:
                weightsPath = './models-2/pong-' + str(episode) + '.pth'
                torch.save(model.state_dict(), weightsPath)
            last_100_reward.append(total_reward)
            row = [episode,step, epsilon, total_reward, total_loss, np.mean(last_100_reward)]
            
            with open(filename,'a') as csvfile:
                #csvwriter = csv.writer(csvfile)
                #csvfile.write(str(row)+'\n')   
                csvfile.write(str(episode)+","+str(step)+","+ str(epsilon)+","+ str(total_reward)+","+ str(total_loss)+","+ str(np.mean(last_100_reward))+"\n")
            print('Episode: %d, #Step %d, Epsilon:%f, Reward: %f, Loss: %f, Last 100 reward: %f' % (episode, step,epsilon, total_reward, total_loss, np.mean(last_100_reward)))
        
            #print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])
            
            break

