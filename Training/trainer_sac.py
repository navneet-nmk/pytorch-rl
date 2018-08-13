# Training script for SAC

import torch
# Add this line to get better performance
torch.backends.cudnn.benchmark=True
from Utils import utils
import torch.optim as optim
from models.SAC import SAC, StochasticActor, Critic, ValueNetwork
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
from osim.env import ProstheticsEnv
import os

if __name__ == '__main__':
    # Specify the environment name and create the appropriate environment
    seed = 4240
    env = ProstheticsEnv(visualize=False)
    eval_env = ProstheticsEnv(visualize=False)

    action_dim = env.get_action_space_size()
    observation_dim = env.get_observation_space_size()

    buffer_capacity = int(1e3)
    q_dim = 1
    v_dim = 1
    batch_size = 128
    hidden_units = 256
    gamma = 0.98  # Discount Factor for future rewards
    num_epochs = 50
    learning_rate = 1e-2
    critic_learning_rate = 1e-2
    value_learning_rate = 1e-2
    polyak_factor = 0.05
    # Adam Optimizer
    opt = optim.Adam

    # Output Folder
    output_folder = os.getcwd() + '/output_sac/'

    # Convert the observation and action dimension to int
    print(observation_dim)
    observation_dim = int(observation_dim)
    action_dim = int(action_dim)
    print(action_dim)

    # Agent definition
    actor = StochasticActor(state_dim=observation_dim, action_dim=action_dim,
                            hidden_dim=hidden_units, use_sigmoid=True)
    critic = Critic(state_dim=observation_dim, action_dim=action_dim,
                    output_dim=q_dim, hidden_dim=hidden_units)
    value = ValueNetwork(state_dim=observation_dim, hidden_dim=hidden_units,
                         output_dim=v_dim)
    target_value = ValueNetwork(state_dim=observation_dim, hidden_dim=hidden_units,
                         output_dim=v_dim)
    sac = SAC(state_dim=observation_dim, action_dim=action_dim,
              hidden_dim=hidden_units, actor=actor, critic=critic,
              value_network=value, actor_learning_rate=learning_rate,
              critic_learning_rate=critic_learning_rate, value_learning_rate=value_learning_rate,
              batch_size=batch_size, buffer_capacity=buffer_capacity, env=env, eval_env=eval_env,
              gamma=gamma, max_episodes_per_epoch=50, nb_train_steps=50, num_epochs=num_epochs,
              num_eval_rollouts=50, num_q_value=q_dim, num_v_value=v_dim, num_rollouts=100,
              output_folder=output_folder, polyak_constant=polyak_factor, random_seed=seed,
              target_value_network=target_value, use_cuda=use_cuda)

    # Train
    sac.train()
