# Training file for the DQN

import gym
import torch
from torch.autograd import Variable
import torch.nn as nn
from models import DQN
import random
import numpy as np
import torch.optim as optim
from itertools import count
import math
import torch.nn.functional as F
# Constants for training
use_cuda = torch.cuda.is_available()
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Preprocessing
steps_done = 0


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    img = downsample(img)
    return img.astype(np.float)


def choose_best_action(model, state):
    state = Variable(torch.FloatTensor(state))
    if use_cuda:
        state = state.cuda()
        model = model.cuda()
    state = state.unsqueeze(0)
    state = torch.transpose(state, 1, 3)
    state = torch.transpose(state, 2, 3)
    Q_values = model(state)
    value, indice = Q_values.max(1)
    action = indice.data[0]
    return action


def get_epsilon_iteration(steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    return eps_threshold


def fit_batch(target_dqn_model, dqn_model, buffer, batch_size, gamma, n, criterion,
              iteration, learning_rate, use_polyak_averaging=True, polyak_constant=0.001):

    # Step 1: Sample mini batch from B uniformly
    if buffer.get_buffer_size() < batch_size:
        return 0, 0
    batch = buffer.sample_batch(batch_size)
    states = []
    new_states = []
    actions = []
    rewards = []
    for k in batch:
        state, action, new_state, reward = k
        states.append(state)
        actions.append(action)
        new_states.append(new_state)
        rewards.append(reward)

    states = torch.FloatTensor(states)
    states = torch.transpose(states, 1, 3)
    states = torch.transpose(states, 2, 3)
    states = Variable(states)
    new_states = torch.FloatTensor(new_states)
    new_states = torch.transpose(new_states, 1, 3)
    new_states = torch.transpose(new_states, 2, 3)
    new_states = Variable(new_states)

    rewards = torch.FloatTensor(rewards)
    rewards = Variable(rewards)

    actions = torch.LongTensor(actions)
    actions = actions.view(-1, 1)
    actions = Variable(actions)

    if use_cuda:
        states = states.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        new_states = new_states.cuda()
        target_dqn_model = target_dqn_model.cuda()
        dqn_model = dqn_model.cuda()

    for p in target_dqn_model.parameters():
        p.requires_grad = False

    # Step 2: Compute the target values using the target network
    Q_values = target_dqn_model(new_states)
    next_Q_values, indice = Q_values.max(1)
    y = rewards + gamma*next_Q_values
    y = y.detach()

    model_parameters = dqn_model.parameters()

    optimizer = optim.Adam(model_parameters, lr=learning_rate)

    # Zero the optimizer gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = dqn_model(states)
    outputs = outputs.gather(1, actions)
    loss = criterion(outputs, y)
    loss.backward()
    # Gradient clipping
    for p in dqn_model.parameters():
        p.grad.data.clamp(-1,1)
    optimizer.step()
    # Stabilizes training as proposed in the DDPG paper
    if use_polyak_averaging:
        t = polyak_constant
        target_dqn_model.conv1.weight.data = t*(dqn_model.conv1.weight.data) + \
                                             (1-t)*(target_dqn_model.conv1.weight.data)
        target_dqn_model.bn1.weight.data = t * (dqn_model.bn1.weight.data) + \
                                             (1 - t) * (target_dqn_model.bn1.weight.data)
        target_dqn_model.conv2.weight.data = t * (dqn_model.conv2.weight.data) + \
                                             (1 - t) * (target_dqn_model.conv2.weight.data)
        target_dqn_model.bn2.weight.data = t * (dqn_model.bn2.weight.data) + \
                                             (1 - t) * (target_dqn_model.bn2.weight.data)
        target_dqn_model.conv3.weight.data = t * (dqn_model.conv3.weight.data) + \
                                             (1 - t) * (target_dqn_model.conv3.weight.data)
        target_dqn_model.bn3.weight.data = t * (dqn_model.bn3.weight.data) + \
                                             (1 - t) * (target_dqn_model.bn3.weight.data)
        target_dqn_model.fully_connected_layer.weight.data = t * (dqn_model.fully_connected_layer.weight.data) + \
                                             (1 - t) * (target_dqn_model.fully_connected_layer.weight.data)
        target_dqn_model.output_layer.weight.data = t * (dqn_model.output_layer.weight.data) + \
                                             (1 - t) * (target_dqn_model.output_layer.weight.data)
    else:
        if n == iteration:
            target_dqn_model.load_state_dict(dqn_model.state_dict())

    return loss, torch.sum(rewards)


def train(target_dqn_model, dqn_model, buffer, batch_size, gamma, n, num_epochs, criterion, learning_rate,
          use_double_q_learning = False):
    for iteration in range(num_epochs):
        print("Epoch ", iteration)
        state = env.reset()
        state = preprocess(state)
        loss = 0
        re = 0
        # Populate the buffer
        for t in count():
            global steps_done
            epsilon = get_epsilon_iteration(steps_done)
            steps_done +=1
            # Choose a random action
            if random.random() < epsilon:
                action = env.action_space.sample()
                new_state, reward, done, info = env.step(action)
            else:
                if use_double_q_learning:
                    action = choose_best_action(dqn_model, state)
                else:
                    action = choose_best_action(target_dqn_model, state)
                new_state, reward, done, info = env.step(action)

            new_state = preprocess(new_state)
            buffer.add((state, action, new_state, reward))
            state = new_state
            # Fit the model on a batch of data
            loss_n, r = fit_batch(target_dqn_model, dqn_model, buffer, batch_size, gamma, n, criterion, iteration, learning_rate)
            #print(loss)
            loss += loss_n
            re += r
            if done:
                break
        print("Loss for episode", iteration, " is ", loss.data/t)
        print("Reward for episode", iteration, " is ", re)

    return target_dqn_model, dqn_model


if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4')
    input_shape = env.observation_space.shape
    img_height, img_width, img_channels = input_shape
    num_actions = env.action_space.n

    dqn_model = DQN.ActionPredictionNetwork(num_conv_layers=16, input_channels=img_channels,
                                            output_q_value=num_actions, pool_kernel_size=3,
                                            kernel_size=3, dense_layer_features=256,
                                            IM_HEIGHT=img_height//2, IM_WIDTH=img_width//2)

    target_dqn_model = DQN.ActionPredictionNetwork(num_conv_layers=16, input_channels=img_channels,
                                            output_q_value=num_actions, pool_kernel_size=3,
                                            kernel_size=3, dense_layer_features=256,
                                            IM_HEIGHT=img_height//2, IM_WIDTH=img_width//2)

    buffer = DQN.ReplayBuffer(size_of_buffer=10000) # Experience Replay
    batch_size= 32
    gamma = 0.99 # Discount factor
    num_epochs = 1000
    learning_rate = 0.01
    # Huber loss to aid small gradients
    criterion = F.smooth_l1_loss
    n = 10 # Target network parameter update
    if use_cuda:
        target_dqn_model = target_dqn_model.cuda()
        dqn_model = dqn_model.cuda()
    model, _ = train(target_dqn_model, dqn_model, buffer, batch_size, gamma, n, num_epochs, criterion, learning_rate,
                     use_double_q_learning=True)
    # Saving the model
    path = '/home/kumar/anaconda3/bin/python /home/kumar/PycharmProjects/Deep-Q-Learning/'
    torch.save(model.state_dict(), path)
















