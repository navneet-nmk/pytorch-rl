# Deep Reinforcement Learning in Pytorch
<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg">


<table>
  <tr>
    <td><img src="/assets/r_her.gif?raw=true" width="200"></td>
    <td><img src="/assets/goal-3.png?raw=true" width="200"></td>
    <td><img src="/assets/virtual-goal.png?raw=true" width="200"></td>
  </tr>
</table>

This repository contains all standard model-free and model-based(coming) RL algorithms in Pytorch. (May also contain some research ideas I am working on currently)

# What is it?
pytorch-rl implements some state-of-the art deep reinforcement learning algorithms in Pytorch, especially those concerned with continuous action spaces. You can train your algorithm efficiently either on CPU or GPU. Furthermore, pytorch-rl works with OpenAI Gym out of the box. This means that evaluating and playing around with different algorithms is easy. Of course you can extend pytorch-rl according to your own needs.
TL:DR : pytorch-rl makes it really easy to run state-of-the-art deep reinforcement learning algorithms.

# Installation

Install Pytorch-rl from Pypi (recommended):

pip install pytorch-policy

# Dependencies
1. Pytorch
2. Gym (OpenAI)
3. mujoco-py (For the physics simulation and the robotics env in gym)
4. Pybullet (Coming Soon)
5. MPI (Only supported with mpi backend Pytorch installation)
6. Tensorboardx (https://github.com/lanpa/tensorboardX)

# RL algorithms
1. DQN (with Double Q learning)
2. DDPG 
3. DDPG with HER (For the OpenAI Fetch Environments)
4. Heirarchical Reinforcement Learning
5. Prioritized Experience Replay + DDPG
6. DDPG with Prioritized Hindsight experience replay (Research)
7. Neural Map with A3C (Coming Soon)
8. Rainbow DQN (Coming Soon)
9. PPO (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
10. HER with self attention for goal substitution (Research)
11. A3C (Coming Soon)
12. ACER (Coming Soon)
13. DARLA
14. TDM
15. World Models
16. Soft Actor-Critic

# Environments
1. Breakout 
2. Pong (coming soon)
3. Hand Manipulation Robotic Task
4. Fetch-Reach Robotic Task
5. Hand-Reach Robotic Task 
6. Block Manipulation Robotic Task
7. Montezuma's Revenge (Current Research)
8. Pitfall
9. Gravitar
10. CarRacing
11. Super Mario Bros (Follow instructions to install gym-retro https://github.com/openai/retro)
12. OpenSim Prosthetics Nips Challenge (https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge)

# Environment Modelling (For exploration and domain adaptation)

Multiple GAN training tricks have been used because of the instability in training the generators and discriminators.
Please refer to https://github.com/soumith/ganhacks for more information.

Even after using the tricks, it was really hard to train a GAN to convergence. 
However, after using Spectral Normalization (https://arxiv.org/abs/1802.05957) the infogan was trained to convergence.

For image to image translation tasks with GANs and for VAEs in general, training with Skip Connection really helps the training.

1. beta-VAE
2. InfoGAN
3. CVAE-GAN
4. Flow based generative models (Research)
5. SAGAN
6. Sequential Attend, Infer, Repeat
7. Curiosity driven exploration
6. Parameter Space Noise for Exploration
7. Noisy Network

# References
1. Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
2. Human-level control through deep reinforcement learning, Mnih et al., 2015
3. Deep Reinforcement Learning with Double Q-learning, van Hasselt et al., 2015
4. Continuous control with deep reinforcement learning, Lillicrap et al., 2015
5. CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training, Bao et al., 2017
6. beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, Higgins et al., 2017
7. Hindsight Experience Replay, Andrychowicz et al., 2017
8. InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets, Chen et al., 2016
9. World Models, Ha et al., 2018
10. Spectral Normalization for Generative Adversarial Networks, Miyato et al., 2018
11. Self-Attention Generative Adversarial Networks, Zhang et al., 2018
12. Curiosity-driven Exploration by Self-supervised Prediction, Pathak et al., 2017
13. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al., 2018
14. Parameter Space Noise for Exploration, Plappert et al., 2018
15. Noisy Network for Exploration, Fortunato et al., 2018
16. Proximal Policy Optimization Algorithms, Schulman et al., 2017
17. Unsupervised Real-Time Control through Variational Empowerment, Karl et al., 2017
18. Mutual Information Neural Estimation, Belghazi et al., 2018
19. Empowerment-driven Exploration using Mutual Information Estimation, Kumar et al., 2018
