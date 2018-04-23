# Training script for the DDPG

import torch
# Add this line to get better performance
torch.backends.cudnn.benchmark=True
from Utils import utils
import torch.optim as optim
from models.DDPG import DDPG
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
from Training.trainer import Trainer
import os

if __name__ == '__main__':
    # Specify the environment name and create the appropriate environment
    seed = 4240
    env = utils.EnvGenerator(name='FetchReach-v1', goal_based=False, seed=seed)
    eval_env = utils.EnvGenerator(name='FetchReach-v1', goal_based=False,seed=seed)
    action_dim = env.get_action_dim()
    observation_dim = env.get_observation_dim()
    goal_dim =  env.get_goal_dim()
    env= env.get_environment()
    eval_env = eval_env.get_environment()

    # Training constants
    her_training=True
    # Future framnes to look at
    future= 4

    buffer_capacity = int(1e3)
    q_dim = 1
    batch_size = 128
    hidden_units = 256
    gamma = 0.98  # Discount Factor for future rewards
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
    print(observation_dim)
    observation_dim = int(observation_dim)
    action_dim = int(action_dim)
    print(action_dim)
    goal_dim= int(goal_dim)

    # Create the agent
    agent = DDPG(num_hidden_units=hidden_units, input_dim=observation_dim+goal_dim,
                      num_actions=action_dim, num_q_val=q_dim, batch_size=batch_size, random_seed=seed,
                      use_cuda=use_cuda, gamma=gamma, actor_optimizer=opt, critic_optimizer=optim,
                      actor_learning_rate=learning_rate, critic_learning_rate=critic_learning_rate,
                      loss_function=criterion, polyak_constant=polyak_factor, buffer_capacity=buffer_capacity,
                 goal_dim=goal_dim, observation_dim=observation_dim)

    # Train the agent
    trainer = Trainer(agent=agent, num_epochs=50, num_rollouts=19*50, num_eval_rollouts=100,
                      max_episodes_per_epoch=50, env=env, eval_env=None,
                      nb_train_steps=19*50, multi_gpu_training=False, random_seed=seed, future=future)

    if her_training:
        trainer.her_training()
    else:
        trainer.train()



