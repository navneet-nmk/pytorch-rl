# Training script for the DDPG

import torch
import utils
import torch.optim as optim
from DDPG import DDPG
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
from trainer import Trainer
import os

if __name__ == '__main__':
    # Specify the environment name and create the appropriate environment
    env = utils.EnvGenerator(name='FetchReach-v1', goal_based=True)
    eval_env = utils.EnvGenerator(name='FetchReach-v1', goal_based=True)
    action_dim = env.get_action_dim()
    observation_dim = env.get_observation_dim()

    # Training constants
    buffer_capacity =  1000000
    q_dim = 1
    batch_size = 256
    hidden_units = 256
    gamma = 0.99  # Discount Factor for future rewards
    num_epochs = 50
    learning_rate = 0.001
    critic_learning_rate = 0.001
    polyak_factor = 0.05
    # Huber loss to aid small gradients
    criterion = F.smooth_l1_loss
    # Adam Optimizer
    opt = optim.Adam

    # Output Folder
    output_folder = os.getcwd() + '/output_ddpg/'

    # Convert the observation and action dimension to int
    observation_dim = int(observation_dim)
    action_dim = int(action_dim)

    # Create the agent
    agent = DDPG(num_hidden_units=hidden_units, input_dim=observation_dim,
                      num_actions=action_dim, num_q_val=q_dim, batch_size=batch_size,
                      use_cuda=use_cuda, gamma=gamma, actor_optimizer=opt, critic_optimizer=optim,
                      actor_learning_rate=learning_rate, critic_learning_rate=critic_learning_rate,
                      loss_function=criterion, polyak_constant=polyak_factor, buffer_capacity=buffer_capacity)

    # Train the agent
    trainer = Trainer(agent=agent, num_epochs=50, num_rollouts=200, num_eval_rollouts=100,
                      max_episodes_per_epoch=1900, env=env, eval_env=eval_env,
                      nb_train_steps=100, multi_gpu_training=False)

    trainer.train()



