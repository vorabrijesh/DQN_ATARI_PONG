from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

MODEL_PATH = './models/pong-'
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 50000000
batch_size = 32
gamma = 0.99
record_idx = 10000

replay_initial = 10000
replay_buffer = ReplayBuffer(50000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load("model_pretrained.pth", map_location='cpu'))

target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_model.copy_from(model)
# target_model.eval()

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

state = env.reset()
fields = ['Frame',"Epsilon","Loss","Reward", "Last 10 Rewards"]
filename = "data6.csv"
for frame_idx in range(1, num_frames+1):
    #print("Frame: " + str(frame_idx))
    #epsilon = epsilon_by_frame(frame_idx)
    #print(epsilon)
    action = model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        if epsilon > epsilon_final:
            epsilon *= 0.99
        # print(episode_reward)
        all_rewards.append((frame_idx, episode_reward))
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        optimizer.zero_grad()
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()))

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses, 0)[1]))
        print('Last-20 average reward: %f' % np.mean(all_rewards[-20:], 0)[1])
        with open(filename, 'a') as csvfile:
            csvfile.write(str(frame_idx)+","+str(epsilon)+","+str(np.mean(losses,0)[1])+","+str(all_rewards[-1][1])+","+str(np.mean(all_rewards[-20:],0)[1])+"\n")
            
    if frame_idx % 20000 == 0:
        target_model.copy_from(model)
        weightsPath = './models-6/pong-' + str(frame_idx) + '.pth'
        torch.save(model.state_dict(), weightsPath)
        # weightsPath = MODEL_PATH + str(frame_idx) + '.pth'
        # torch.save(target_model.state_dict(), weightsPath)


