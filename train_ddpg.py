# Training script for the DDPG

import gym
import torch
from torch.autograd import Variable
import torch.nn as nn
import DQN
import random
import numpy as np
import torch.optim as optim
from itertools import count
import math
import DDPG
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


def get_epsilon_iteration(steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    return eps_threshold

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


def fit_batch(target_actor, actor, target_critic, critic, buffer, batch_size, gamma, n, criterion,
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
        target_actor = target_actor.cuda()
        target_critic = target_critic.cuda()
        actor = actor.cuda()
        citic = critic.cuda()

    for p in target_actor.parameters():
        p.requires_grad = False

    for p in target_critic.parameters():
        p.requires_grad = False

    # Step 2: Compute the target values using the target actor network and target critic network
    # Compute the Q-values given the current state ( in this case it is the new_states)
    Q_values = target_critic(new_states)
    # Find the Q-value for the action according to the target actior network
    # We do this because caluclating max over a continuous action space is intractable
    action_taken  = target_actor(new_states)
    next_Q_values = Q_values[action_taken]
    y = rewards + gamma*next_Q_values
    y = y.detach()

    actor_parameters = actor.parameters()
    critic_parameters = critic.parameters()

    optimizer_actor = optim.Adam(actor_parameters, lr=learning_rate)
    optimizer_critic = optim.Adam(critic_parameters, lr=learning_rate)

    # Zero the optimizer gradients
    optimizer_critic.zero_grad()

    # Forward pass
    outputs = critic(states)
    outputs = outputs.gather(1, actions)
    loss = criterion(outputs, y)
    loss.backward()
    # Gradient clipping
    for p in critic.parameters():
        p.grad.data.clamp(-1,1)
    optimizer_critic.step()

    # Updating the actor policy


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


def train(target_actor, actor, target_critic, critic,  buffer, batch_size, gamma, n, num_epochs, criterion, learning_rate):
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
                # Action is taken by the actor network u
                action = actor(state)
                new_state, reward, done, info = env.step(action)

            new_state = preprocess(new_state)
            buffer.add((state, action, new_state, reward))
            state = new_state
            # Fit the model on a batch of data
            loss_n, r = fit_batch(target_actor, actor,  target_critic, critic, buffer, batch_size, gamma, n, criterion, iteration, learning_rate)
            #print(loss)
            loss += loss_n
            re += r
            if done:
                break
        print("Loss for episode", iteration, " is ", loss.data/t)
        print("Reward for episode", iteration, " is ", re)

    return target_actor, target_critic, actor, critic



if __name__ == '__main__':
    # Specify the environment and the corresponding dimensions
    env = gym.make('HandManipulateBlock-v0')
    #env.reset()
    #env.render()
    print(env.observation_space)
    print(env.action_space.shape)
    obs_space = env.observation_space
    print(obs_space)
    achieved_goal = obs_space['achieved_goal']
    desired_goal = obs_space['desired_goal']
    observation = obs_space['observation']
    print(achieved_goal, desired_goal, observation)
    input_shape = env.observation_space.shape
    #print(input_shape)
    img_height, img_width, img_channels = input_shape
    num_actions = env.action_space.n

    print(input_shape, " ", num_actions)

    num_q_value= 1
    # We need 4 networks
    # Target Actor - Takes the state as input
    target_actor = DDPG.ActorDDPGNetwork(num_conv_layers=32, conv_kernel_size=3, input_channels=img_channels, output_action=num_actions,
                                         IMG_HEIGHT=img_height, IMG_WIDTH=img_width, pool_kernel_size=2)
    actor = DDPG.ActorDDPGNetwork(num_conv_layers=32, conv_kernel_size=3, input_channels=img_channels, output_action=num_actions,
                                  IMG_HEIGHT=img_height, IMG_WIDTH=img_width, pool_kernel_size=2)

    # Target Critic - Takes the state and action as input
    target_critic = DDPG.CriticDDPGNetwork(num_conv_layers = 32, conv_kernel_size=3, input_channels=img_channels, output_q_value=num_q_value,
                                           IMG_HEIGHT=img_height, IMG_WIDTH=img_width, pool_kernel_size=2)
    critic = DDPG.CriticDDPGNetwork(num_conv_layers=32, conv_kernel_size=3, input_channels=img_channels, output_q_value=num_q_value,
                                    IMG_HEIGHT=img_height, IMG_WIDTH=img_width, pool_kernel_size=2)

