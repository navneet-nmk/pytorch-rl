import gym
import torch
from torch.autograd import Variable
import torch.nn as nn
import model
import random
import numpy as np
import torch.optim as optim
from itertools import count


# Preprocessing

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    img = downsample(img)
    return img.astype(np.float)

def choose_best_action(model, state):
    state = Variable(torch.FloatTensor(state))
    state = state.unsqueeze(0)
    state = torch.transpose(state, 1, 3)
    state = torch.transpose(state, 2, 3)
    Q_values = model(state)
    value, indice = Q_values.max(1)
    action = indice.data[0]
    return action


def get_epsilon_iteration(iteration):
    if iteration < 10:
        return 0.5
    elif iteration < 100:
        return 0.3
    else:
        return 0.1


def fit_batch(dqn_model, buffer, batch_size, gamma, n, criterion, iteration, learning_rate):

    # Step 1: Sample mini batch from B uniformly
    if buffer.get_buffer_size() < batch_size:
        return 0
    batch = buffer.sample_batch(batch_size)
    states = []
    new_states = []
    actions = []
    rewards = []
    for k in batch:
        state, action, new_state, reward = k
        states.append(state)
        actions.append(action)
        new_states.append(new_states)
        rewards.append(reward)

    states = torch.FloatTensor(states)
    states = torch.transpose(states, 1, 3)
    states = torch.transpose(states, 2, 3)
    states = Variable(states)

    rewards = torch.FloatTensor(rewards)
    rewards = Variable(rewards)

    actions= torch.LongTensor(actions)
    actions = actions.view(-1, 1)
    actions = Variable(actions)

    # Step 2: Compute the target values using the target network
    Q_values = dqn_model(states)
    next_Q_values, indice = Q_values.max(1)
    y = rewards + gamma*next_Q_values
    y = y.detach()
    # Step 3:
    #target_network_parameters = dqn_model.parameters()
    #for p in target_network_parameters:
     #   p.requires_grad = False

    model_parameters = dqn_model.parameters()

    optimizer = optim.Adam(model_parameters, lr=learning_rate)

    # Zero the optimizer gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = dqn_model(states)
    outputs = outputs.gather(1, actions)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    return loss


def train(dqn_model, buffer, batch_size, gamma, n, num_epochs, criterion, learning_rate):
    for iteration in range(num_epochs):
        state = env.reset()
        state = preprocess(state)
        loss =0
        # Populate the buffer
        for t in count():
            epsilon = get_epsilon_iteration(iteration)
            # Choose a random action
            if random.random() < epsilon:
                action = env.action_space.sample()
                new_state, reward, done, info = env.step(action)
            else:
                action = choose_best_action(dqn_model, state)
                new_state, reward, done, info = env.step(action)

            new_state = preprocess(new_state)
            buffer.add((state, action, new_state, reward))
            state = new_state
            # Fit the model on a batch of data
            loss += fit_batch(dqn_model, buffer, batch_size, gamma, n, criterion, iteration, learning_rate)
            if done:
                break
        print(loss/t)
        print("\n")


if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4')
    input_shape = env.observation_space.shape
    img_height, img_width, img_channels = input_shape
    num_actions = env.action_space.n

    dqn_model = model.ActionPredictionNetwork(num_conv_layers=16, input_channels=img_channels,
                                              output_q_value=num_actions, pool_kernel_size=3,
                                              kernel_size=3,
                                              IM_HEIGHT=img_height//2, IM_WIDTH=img_width//2)

    buffer = model.ReplayBuffer(size_of_buffer=1000)
    batch_size= 32
    gamma = 0.99
    num_epochs = 40
    learning_rate = 0.01
    criterion = nn.MSELoss()
    n = 5
    train(dqn_model, buffer, batch_size, gamma, n, num_epochs, criterion, learning_rate)

















