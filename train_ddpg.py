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


def polyak_update(polyak_factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy(
            polyak_factor*param + target_param*(1.0 - polyak_factor)
        )


def fit_batch(target_actor, actor, target_critic, critic, buffer, batch_size, gamma, n, criterion,
              iteration, learning_rate, use_polyak_averaging=True, polyak_constant=0.9):

    # Step 1: Sample mini batch from B uniformly
    if buffer.get_buffer_size() < batch_size:
        # If buffer is still not full enough return 0
        return 0, 0
    # Sample a minibatch from the buffer
    transitions = buffer.sample_batch(batch_size)
    batch = Buffer.Transition(*zip(*transitions))
    states = []
    new_states = []
    actions = []
    rewards = []
    achieved_goals = []
    desired_goals = []
    new_achieved_goals = []
    new_desired_goals = []
    successes =[]

    states = batch.state
    new_states = batch.next_state
    actions = batch.action
    rewards = batch.reward
    achieved_goals = batch.achieved_goal
    desired_goals = batch.desired_goal
    successes = batch.success


    # Compute a mask of non final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    print(non_final_mask)


    states = torch.FloatTensor(states)
    states = Variable(states)
    new_states = torch.FloatTensor(new_states)
    new_states = Variable(new_states)

    rewards = torch.FloatTensor(rewards)
    rewards = Variable(rewards)

    actions = torch.LongTensor(actions)
    actions = Variable(actions)

    if use_cuda:
        states = states.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        new_states = new_states.cuda()
        target_actor = target_actor.cuda()
        target_critic = target_critic.cuda()
        actor = actor.cuda()
        critic = critic.cuda()

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
    y = rewards + gamma*next_Q_values
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
        p.grad.data.clamp(-1,1)
    optimizer_critic.step()

    # Updating the actor policy
    policy_loss = -critic(states, actor(states))
    policy_loss=  policy_loss.mean()
    policy_loss.backward()
    for p in actor.parameters():
        p.grad.data.clamp(-1,1)
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

    return loss, torch.sum(rewards)


def train(target_actor, actor, target_critic, critic,  buffer, batch_size, gamma, n,
          num_epochs, criterion, learning_rate, her_training=False):
    for iteration in range(num_epochs):
        print("Epoch ", iteration)
        vector = env.reset()
        state = vector['observation']
        achieved_goal = vector['achieved_goal']
        desired_goal = vector['desired_goal']
        #state = preprocess(state)
        loss = 0
        re = 0
        # Populate the buffer
        t = 0
        success_n = 0
        for t in count():
            global steps_done
            epsilon = get_epsilon_iteration(steps_done)
            steps_done +=1
            # Choose a random action
            if random.random() < epsilon:
                action = env.action_space.sample()

            else:
                # Action is taken by the actor network

                state = torch.FloatTensor(state)
                if use_cuda:
                    state = state.cuda()

                print(state)
                state = Variable(state)
                action = actor(state)
                action = action.data.cpu().numpy()[0]
                print(action)

            new_vector, reward, done, successes = env.step(action)
            print(new_vector)
            new_state = new_vector['observation']
            new_acheived_goal = new_vector['achieved_goal']
            new_desired_goal = new_vector['desired_goal']
            success = successes['is_success']

            #new_state = preprocess(new_state)
            if her_training:
                buffer.pusj((state, action, new_state, reward, achieved_goal,
                            desired_goal, new_acheived_goal, new_desired_goal, sucess))
            else:
                buffer.push(state, action, new_state, reward, achieved_goal, desired_goal, new_acheived_goal, new_desired_goal, success)
            state = new_state
            # Fit the model on a batch of data
            loss_n, r = fit_batch(target_actor, actor,  target_critic, critic, buffer,
                                  batch_size, gamma, n, criterion, iteration, learning_rate)
            loss += loss_n
            re += r
            success_n += success
            if done:
                break
        if t > 0:
            loss_calc = loss.data/t
            re = re/t
            success_n = success_n/t
        else:
            loss_calc = loss.data
        print("Loss for episode", iteration, " is ", loss_calc)
        print("Reward for episode", iteration, " is ", re)
        print("Success Rate for the episode ", iteration, " is " ,success_n)

    return target_actor, target_critic, actor, critic



if __name__ == '__main__':
    # Specify the environment and the corresponding dimensions
    env = gym.make('HandManipulateBlock-v0')
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
    print(input_shape, " ", num_actions)
    action = env.action_space.sample()
    #print(action)

    num_q_value= 1
    # We need 4 networks
    # Initialize the actor and critic networks
    actor = DDPG.ActorDDPGNonConvNetwork(num_hidden_layers=64, output_action=num_actions, input=input_shape)
    critic = DDPG.CriticDDPGNonConvNetwork(num_hidden_layers=64, output_q_value=num_q_value, input=input_shape)
    # Initialize the target actor and target critic networks
    target_actor = DDPG.ActorDDPGNonConvNetwork(num_hidden_layers=64, output_action=num_actions, input=input_shape)
    target_critic = DDPG.CriticDDPGNonConvNetwork(num_hidden_layers=64, output_q_value=num_q_value, input=input_shape)
    # Set the weights of the target networks similar to the general networks
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    # Initialize the replay buffer
    buffer = Buffer.ReplayBuffer(capacity=10000)
    batch_size = 64
    gamma = 0.99 # Discount Factor for future rewards
    num_epochs = 1000
    learning_rate = 0.1
    # Huber loss to aid small gradients
    criterion = F.smooth_l1_loss
    # Target network parameter update if not using polyak averaging
    n = 20
    if use_cuda:
        target_actor = target_actor.cuda()
        target_critic = target_critic.cuda()
        actor = actor.cuda()
        critic = critic.cuda()

    t_actor, t_critic, ac, cr = train(target_actor=target_actor, target_critic=target_critic, actor=actor,
                                      critic=critic, buffer=buffer, batch_size=batch_size, gamma=gamma,
                                      n=n, num_epochs=num_epochs, criterion=criterion, learning_rate=learning_rate)

