# Training script for the DDPG

import gym
import torch
from torch.autograd import Variable
import random
import numpy as np
import torch.optim as optim
from itertools import count
import math
import DDPG
import torch.nn.functional as F
import Buffer
import torch.nn as nn
# Constants for training
use_cuda = torch.cuda.is_available()
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500

# Preprocessing
steps_done = 0


def compute_td_loss(batch_size):
    # Sample a minibatch from the buffer
    transitions = buffer.sample_batch(batch_size)
    batch = Buffer.Transition(*zip(*transitions))

    states = batch.state
    new_states = batch.next_state
    actions = batch.action
    rewards = batch.reward
    achieved_goals = batch.achieved_goal
    desired_goals = batch.desired_goal
    new_achieved_goals = batch.new_achieved_goal
    new_desired_goals = batch.new_desired_goal
    successes = batch.success
    dones = batch.done

    states = Variable(torch.cat(states))
    new_states = Variable(torch.cat(new_states), volatile=True)
    actions = Variable(torch.cat(actions))
    rewards = Variable(torch.cat(rewards))
    dones = Variable(torch.cat(dones))

    if use_cuda:
        states = states.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        new_states = new_states.cuda()
        dones = dones.cuda()


    # Get the next action to take according to the next states
    next_actions = target_actor(new_states)
    # Compute the next q values given the next states and next actions
    # Find the Q-value for the action according to the target actor network
    # We do this because caluclating max over a continuous action space is intractable
    next_q_values = target_critic(new_states, next_actions)
    # Calculate the target value for the bellman update
    y = rewards + gamma*next_q_values*(1-dones)
    y = y.detach()

    actor_parameters = actor.parameters()
    critic_parameters = critic.parameters()

    optimizer_actor = optim.Adam(actor_parameters, lr=learning_rate)
    optimizer_critic = optim.Adam(critic_parameters, lr=learning_rate)

    # Zero the optimizer gradients
    optimizer_critic.zero_grad()

    # Forward pass
    outputs = critic(states, actions)
    loss = criterion(outputs, y)
    loss.backward()
    # Gradient clipping
    for p in critic.parameters():
        p.grad.data.clamp(-1, 1)
    optimizer_critic.step()

    # Updating the actor policy
    policy_loss = -critic(states, actor(states))
    policy_loss = policy_loss.mean()

    policy_loss.backward()
    optimizer_actor.step()

    return loss





def get_epsilon_iteration(steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    return eps_threshold

def polyak_update(polyak_factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy = polyak_factor*param + target_param*(1.0 - polyak_factor)



def fit_batch(target_actor, actor, target_critic, critic, buffer, batch_size, gamma, n, criterion,
              iteration, learning_rate, critic_learning_rate, use_polyak_averaging=True, polyak_constant=0.001):

    # Step 1: Sample mini batch from B uniformly
    if buffer.get_buffer_size() < batch_size:
        # If buffer is still not full enough return 0
        return 0
    # Sample a minibatch from the buffer
    transitions = buffer.sample_batch(batch_size)
    batch = Buffer.Transition(*zip(*transitions))

    states = batch.state
    new_states = batch.next_state
    actions = batch.action
    rewards = batch.reward
    achieved_goals = batch.achieved_goal
    desired_goals = batch.desired_goal
    new_achieved_goals = batch.new_achieved_goal
    new_desired_goals = batch.new_desired_goal
    successes = batch.success
    dones = batch.done

    states = Variable(torch.cat(states))
    new_states = Variable(torch.cat(new_states))
    actions = Variable(torch.cat(actions))
    rewards =  Variable(torch.cat(rewards))
    dones = Variable(torch.cat(dones))

    #print(states.shape)
    #print(new_states.shape)
    #print(actions.shape)
    #print(rewards.shape)

    if use_cuda:
        states = states.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        new_states = new_states.cuda()
        dones = dones.cuda()

    for p in target_actor.parameters():
        p.requires_grad = False

    for p in target_critic.parameters():
        p.requires_grad = False

    # Step 2: Compute the target values using the target actor network and target critic network
    # Compute the Q-values given the current state ( in this case it is the new_states)
    new_action = target_actor(new_states)
    next_Q_values = target_critic(new_states, new_action)
    # Find the Q-value for the action according to the target actior network
    # We do this because caluclating max over a continuous action space is intractable
    next_Q_values = torch.squeeze(next_Q_values, dim=1)
    next_Q_values = next_Q_values * (1-dones)
    m = gamma*next_Q_values
    #print(rewards)
    #rewards = torch.squeeze(rewards, dim=1)
    y = rewards + m
    y = y.detach()

    actor_parameters = actor.parameters()
    critic_parameters = critic.parameters()

    optimizer_actor = optim.Adam(actor_parameters, lr=learning_rate)
    optimizer_critic = optim.Adam(critic_parameters, lr=critic_learning_rate)

    # Zero the optimizer gradients
    optimizer_critic.zero_grad()

    # Forward pass
    outputs = critic(states, actions)
    loss = criterion(outputs, y)
    loss.backward()
    # Gradient clipping
    for p in critic.parameters():
        p.grad.data.clamp(-1,1)
    optimizer_critic.step()

    # Updating the actor policy
    policy_loss = -critic(states, actor(states))
    policy_loss =  policy_loss.mean()

    policy_loss.backward()
    optimizer_actor.step()

    # Stabilizes training as proposed in the DDPG paper
    if use_polyak_averaging:
        t = polyak_constant
        polyak_update(t, target_network=target_actor, network=actor)
        polyak_update(t, target_network=target_critic, network=critic)
    else:
        if n == iteration:
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())

    return loss


def train(target_actor, actor, target_critic, critic,  buffer, batch_size, gamma, n,
          num_epochs, criterion, learning_rate, critic_learning_rate, her_training=False):
    all_rewards = []
    suc = []
    max_eps_steps = 2000
    for iteration in range(num_epochs):
        vector = env.reset()
        state = vector['observation']
        achieved_goal = vector['achieved_goal']
        desired_goal = vector['desired_goal']
        #state = preprocess(state)
        loss = 0
        episode_reward = 0
        success_n = 0
        if use_cuda:
            state = torch.FloatTensor(state)
            with torch.cuda.device(0):
                state = state.cuda()
        else:
            state = torch.FloatTensor(state)

        state =  torch.unsqueeze(state, dim=0)

        for t in range(max_eps_steps):
            global steps_done
            epsilon = get_epsilon_iteration(steps_done)
            steps_done +=1
            # Choose a random action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Action is taken by the actor network

                state_v = Variable(state)
                action = actor(state_v)
                action = action.data.cpu().numpy()[0]


            new_vector, reward, done, successes = env.step(action)
            new_state = new_vector['observation']
            new_acheived_goal = new_vector['achieved_goal']
            new_desired_goal = new_vector['desired_goal']
            success = successes['is_success']
            done_t = done*1

            new_state = torch.cuda.FloatTensor(new_state)
            action = torch.cuda.FloatTensor(action)
            reward = np.asscalar(reward)
            reward = torch.FloatTensor([reward])
            done_t = torch.cuda.FloatTensor([done_t])

            new_state_r =  torch.unsqueeze(new_state, dim=0)
            action_r = torch.unsqueeze(action, dim=0)
#            reward_r = torch.unsqueeze(reward, dim=0)


            #new_state = preprocess(new_state)
            if her_training:
                buffer.push((state, action, new_state_r, reward,  done_t, achieved_goal,
                            desired_goal, new_acheived_goal, new_desired_goal, success))
            else:
                #print(state)
                buffer.push(state, action_r, new_state_r, reward, done_t, achieved_goal, desired_goal, new_acheived_goal, new_desired_goal, success)

            state = new_state

            if use_cuda:
                with torch.cuda.device(0):
                    state = state.cuda()
            state = torch.unsqueeze(state, dim=0)
            # Fit the model on a batch of data
            loss_n = fit_batch(target_actor, actor,  target_critic, critic, buffer,
                                  batch_size, gamma, n, criterion, iteration, learning_rate, critic_learning_rate )
            loss += loss_n
            episode_reward += reward
            success_n += success

            if done:
                all_rewards.append(episode_reward)
                suc.append(success_n.data[0])
                break

        if iteration % 200 == 0:
            print("Epoch ", iteration)
            print("Reward for episode", iteration, " is ", all_rewards[len(all_rewards) - 1])
            print("Success Rate for the episode ", iteration, " is " ,np.sum(suc))

    return target_actor, target_critic, actor, critic



if __name__ == '__main__':
    # Specify the environment and the corresponding dimensions
    env = gym.make('FetchReach-v0')
    obs_space = env.observation_space
    s = env.reset()
    observation = s['observation']
    achieved_goal = s['achieved_goal']
    desired_goal = s['desired_goal']
    obs_shape = observation.shape
    ach_g_shape = achieved_goal.shape
    des_g_shape = desired_goal.shape
    input_shape = obs_shape[0]
    num_actions = env.action_space.shape[0]
    print("Input dimension ", input_shape, " action space ", num_actions)
    action = env.action_space.sample()
    #print(action)

    num_q_value= 1
    # We need 4 networks
    # Initialize the actor and critic networks
    actor = DDPG.ActorDDPGNonConvNetwork(num_hidden_layers=32, output_action=num_actions, input=input_shape)
    critic = DDPG.CriticDDPGNonConvNetwork(num_hidden_layers=32, output_q_value=num_q_value, input=input_shape)
    # Initialize the target actor and target critic networks
    target_actor = DDPG.ActorDDPGNonConvNetwork(num_hidden_layers=32, output_action=num_actions, input=input_shape)
    target_critic = DDPG.CriticDDPGNonConvNetwork(num_hidden_layers=32, output_q_value=num_q_value, input=input_shape)
    # Set the weights of the target networks similar to the general networks
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    # Initialize the replay buffer
    buffer = Buffer.ReplayBuffer(capacity=100000)
    batch_size = 64
    gamma = 0.99 # Discount Factor for future rewards
    num_epochs = 1000
    learning_rate = 0.001
    critic_learning_rate = 0.001
    # Huber loss to aid small gradients
    criterion = F.smooth_l1_loss
    # Target network parameter update if not using polyak averaging
    n = 20
    if use_cuda:
        target_actor = nn.DataParallel(target_actor).cuda()
        target_critic = nn.DataParallel(target_critic).cuda()
        actor = nn.DataParallel(actor).cuda()
        critic = nn.DataParallel(critic).cuda()

    t_actor, t_critic, ac, cr = train(target_actor=target_actor, target_critic=target_critic, actor=actor,
                                      critic=critic, buffer=buffer, batch_size=batch_size, gamma=gamma,
                                      n=n, num_epochs=num_epochs, criterion=criterion, learning_rate=learning_rate, critic_learning_rate=critic_learning_rate)

