# Deep Reinforcement Learning in Pytorch
This repository contains all standard model-free and model-based(coming) RL algorithms in Pytorch

# What is it?
pytorch-rl implements some state-of-the art deep reinforcement learning algorithms in Python, especially those concerned with continuous action spaces. You can train your algorithm efficiently either on CPU or GPU. Furthermore, pytorch-rl works with OpenAI Gym out of the box. This means that evaluating and playing around with different algorithms is easy. Of course you can extend keras-rl according to your own needs.
In a nutshell: pytorch-rl makes it really easy to run state-of-the-art deep reinforcement learning algorithms.

# Dependencies
1. Pytorch
2. Gym (OpenAI)
3. mujoco-py (For the physics simulation and the robotics env in gym)

# RL algorithms
1. DQN ( with Double Q learning and polyak averaging for target network parameter update)
2. DDPG 
3. DDPG with HER
4. Heirarchical Reinforcement Learning
5. Prioritized Experience Replay + DDPG
6. DDPG with Prioritized Hindsight experience replay (Research)

# Environments
1. Breakout 
2. Pong (coming soon)
3. Hand Manipulation Robotic Task
4. Fetch Reach Robotic Task


# References
1. Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
2. Human-level control through deep reinforcement learning, Mnih et al., 2015
3. Deep Reinforcement Learning with Double Q-learning, van Hasselt et al., 2015
4. Continuous control with deep reinforcement learning, Lillicrap et al., 2015
