# This script contains the Actor and Critic classes
import torch.nn as nn
import math
import torch
import numpy as np
import Utils.random_process as random_process
from torch.autograd import Variable
from Memory import Buffer
import torch.optim as opt


class DDPG(object):
    """
    The Deep Deterministic policy gradient network
    """

    def __init__(self, num_hidden_units, input_dim, num_actions, num_q_val,
                 observation_dim, goal_dim,
                 batch_size, use_cuda, gamma, random_seed,
                 actor_optimizer, critic_optimizer,
                 actor_learning_rate, critic_learning_rate,
                 loss_function, polyak_constant,
                 buffer_capacity,non_conv=True,
                 num_conv_layers=None, num_pool_layers=None,
                 conv_kernel_size=None, img_height=None, img_width=None,
                 input_channels=None):

        self.num_hidden_units = num_hidden_units
        self.non_conv = non_conv
        self.num_actions = num_actions
        self.num_q = num_q_val
        self.obs_dim = observation_dim
        self.goal_dim = goal_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.cuda = use_cuda
        self.gamma = gamma
        self.seed(random_seed)
        self.actor_optim = actor_optimizer
        self.critic_optim = critic_optimizer
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.criterion = loss_function
        self.tau = polyak_constant
        self.buffer = Buffer.ReplayBuffer(capacity=buffer_capacity, seed=random_seed)

        # Convolution Parameters
        self.num_conv = num_conv_layers
        self.pool = num_pool_layers
        self.im_height = img_height
        self.im_width = img_width
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels

        if non_conv:
            self.target_actor = ActorDDPGNonConvNetwork(num_hidden_layers=num_hidden_units,
                                                        output_action=num_actions, input=input_dim)

            self.actor = ActorDDPGNonConvNetwork(num_hidden_layers=num_hidden_units,
                                                        output_action=num_actions, input=input_dim)

            self.target_critic = CriticDDPGNonConvNetwork(num_hidden_layers=num_hidden_units,
                                                          output_q_value=num_q_val, input=input_dim, action_dim=num_actions,
                                                          goal_dim=self.goal_dim)
            self.critic = CriticDDPGNonConvNetwork(num_hidden_layers=num_hidden_units,
                                                          output_q_value=num_q_val, input=input_dim, action_dim=num_actions,
                                                   goal_dim=self.goal_dim)

        else:
            self.target_actor = ActorDDPGNetwork(num_conv_layers=self.num_conv, conv_kernel_size=self.conv_kernel_size,
                                                 input_channels=self.input_channels, output_action=self.num_actions,
                                                 dense_layer=self.num_hidden_units, pool_kernel_size=self.pool,
                                                 IMG_HEIGHT=self.im_height, IMG_WIDTH=self.im_width)

            self.actor = ActorDDPGNetwork(num_conv_layers=self.num_conv, conv_kernel_size=self.conv_kernel_size,
                                                 input_channels=self.input_channels, output_action=self.num_actions,
                                                 dense_layer=self.num_hidden_units, pool_kernel_size=self.pool,
                                                 IMG_HEIGHT=self.im_height, IMG_WIDTH=self.im_width)

            self.target_critic = CriticDDPGNetwork(num_conv_layers=self.num_conv, conv_kernel_size=self.conv_kernel_size,
                                                 input_channels=self.input_channels, output_q_value=self.num_q,
                                                 dense_layer=self.num_hidden_units, pool_kernel_size=self.pool,
                                                 IMG_HEIGHT=self.im_height, IMG_WIDTH=self.im_width)
            self.critic = CriticDDPGNetwork(num_conv_layers=self.num_conv, conv_kernel_size=self.conv_kernel_size,
                                                 input_channels=self.input_channels, output_q_value=self.num_q,
                                                 dense_layer=self.num_hidden_units, pool_kernel_size=self.pool,
                                                 IMG_HEIGHT=self.im_height, IMG_WIDTH=self.im_width)
        if self.cuda:
            self.target_actor = self.target_actor.cuda()
            self.actor = self.actor.cuda()
            self.target_critic = self.target_critic.cuda()
            self.critic = self.critic.cuda()

        # Initializing the target networks with the standard network weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Create the optimizers for the actor and critic using the corresponding learning rate
        actor_parameters = self.actor.parameters()
        critic_parameters = self.critic.parameters()

        self.actor_optim = opt.Adam(actor_parameters, lr=self.actor_lr)
        self.critic_optim = opt.Adam(critic_parameters, lr=self.critic_lr)

        # Initialize a random exploration noise
        self.random_noise = random_process.OrnsteinUhlenbeckActionNoise(self.num_actions)

    def to_cuda(self):
        self.target_actor = self.target_actor.cuda()
        self.target_critic = self.target_critic.cuda()
        self.actor = self.actor.cuda()
        self.critic = self.critic.cuda()

    def save_model(self, output):
        """
        Saving the models
        :param output:
        :return:
        """
        print("Saving the actor and critic")
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def random_action(self):
        """
        Take a random action bounded between min and max values of the action space
        :return:
        """
        action = np.random.uniform(-1., 1., self.num_actions)
        self.a_t = action

        return action

    def seed(self, s):
        """
        Setting the random seed for a particular training iteration
        :param s:
        :return:
        """
        torch.manual_seed(s)
        if self.cuda:
            torch.cuda.manual_seed(s)

    def get_actors(self):
        return {'target': self.target_actor, 'actor': self.actor}

    def get_critics(self):
        return {'target': self.target_critic, 'critic': self.critic}

    # Get the action with an option for random exploration noise
    def get_action(self, state, noise=True):
        state_v = Variable(state)
        action = self.actor(state_v)
        if noise:
            noise = self.random_noise
            action = action.data.cpu().numpy()[0] + noise.sample()
        else:
            action = action.data.cpu().numpy()[0]
        action = np.clip(action, -1., 1.)
        return action

    # Reset the noise
    def reset(self):
        self.random_noise.reset()

    # Store the transition into the replay buffer
    def store_transition(self, state, new_state, action, reward, done, success):
        self.buffer.push(state, action, new_state, reward, done, success)

    # Update the target networks using polyak averaging
    def update_target_networks(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    # Calculate the Temporal Difference Error
    def calc_td_error(self, transition):
        """
        Calculates the td error against the bellman target
        :return:
        """
        # Calculate the TD error only for the particular transition

        # Get the separate values from the named tuple
        state, new_state, reward, success, action, done = transition

        state = Variable(state)
        new_state = Variable(new_state)
        reward = Variable(reward)
        action = Variable(action)
        done = Variable(done)

        if self.cuda:   
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            new_state = new_state.cuda()
            done = done.cuda()

        new_action = self.target_actor(new_state)
        next_Q_value = self.target_critic(new_state, new_action)
        # Find the Q-value for the action according to the target actior network
        # We do this because calculating max over a continuous action space is intractable
        next_Q_value.volatile = False
        next_Q_value = torch.squeeze(next_Q_value, dim=1)
        next_Q_value = next_Q_value * (1 - done)
        y = reward + self.gamma * next_Q_value

        outputs = self.critic(state, action)
        td_loss = self.criterion(outputs, y)
        return td_loss

    # Train the networks
    def fit_batch(self):
        # Sample mini-batch from the buffer uniformly or using prioritized experience replay

        # If the size of the buffer is less than batch size then return
        if self.buffer.get_buffer_size() < self.batch_size:
            return None, None

        transitions = self.buffer.sample_batch(self.batch_size)
        batch = Buffer.Transition(*zip(*transitions))

        # Get the separate values from the named tuple
        states = batch.state
        new_states = batch.next_state
        actions = batch.action
        rewards = batch.reward
        dones = batch.done

        #actions = list(actions)
        rewards = list(rewards)
        dones = list(dones)

        states = Variable(torch.cat(states))
        new_states = Variable(torch.cat(new_states), volatile=True)
        actions = Variable(torch.cat(actions))
        rewards = Variable(torch.cat(rewards))
        dones = Variable(torch.cat(dones))

        if self.cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            new_states = new_states.cuda()
            dones = dones.cuda()

        # Step 2: Compute the target values using the target actor network and target critic network
        # Compute the Q-values given the current state ( in this case it is the new_states)
        #with torch.no_grad():

        new_action = self.target_actor(new_states)
        new_action.volatile = True
        next_Q_values = self.target_critic(new_states, new_action)
        # Find the Q-value for the action according to the target actior network
        # We do this because calculating max over a continuous action space is intractable
        # next_Q_values.volatile = False
        next_Q_values = torch.squeeze(next_Q_values, dim=1)
        next_Q_values = next_Q_values * (1 - dones)
        next_Q_values.volatile = False
        y = rewards + self.gamma*next_Q_values

        # Zero the optimizer gradients
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        # Forward pass
        outputs = self.critic(states, actions)
        loss = self.criterion(outputs, y)
        loss.backward()
        # Clamp the gradients to avoid vanishing gradient problem
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optim.step()

        # Updating the actor policy
        policy_loss = -1 * self.critic(states, self.actor(states))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        # Clamp the gradients to avoid the vanishing gradient problem
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optim.step()

        return loss, policy_loss


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorDDPGNetwork(nn.Module):
    # The actor network takes the state as input and outputs an action
    # The actor network is used to approximate the argmax action in a continous action space
    # The actor network in the case of a discrete action space is just argmax_a(Q(s,a))

    def __init__(self, num_conv_layers, conv_kernel_size, input_channels, output_action, dense_layer,
                 pool_kernel_size, IMG_HEIGHT, IMG_WIDTH):
        super(ActorDDPGNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_kernel = conv_kernel_size
        self.input_channels = input_channels
        self.output_action = output_action
        self.dense_layer = dense_layer
        self.pool_kernel_size = pool_kernel_size
        self.im_height = IMG_HEIGHT
        self.im_width = IMG_WIDTH

        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn3 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu3 = nn.ReLU(inplace=True)

        # Fully connected layer
        self.fully_connected_layer = nn.Linear(234432, self.dense_layer)
        self.relu4 = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(self.dense_layer, output_action)

        # Weight initialization from a uniform gaussian distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        x = self.relu4(x)
        out = self.output_layer(x)
        return out


# For non image state space
class ActorDDPGNonConvNetwork(nn.Module):
    def __init__(self, num_hidden_layers, output_action, input):
        super(ActorDDPGNonConvNetwork, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input = input
        self.output_action = output_action
        self.init_w = 3e-3

        #Dense Block
        self.dense_1 = nn.Linear(self.input, self.num_hidden_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.dense_2 = nn.Linear(self.num_hidden_layers, self.num_hidden_layers)
        self.relu2 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.num_hidden_layers, self.output_action)
        self.tanh = nn.Tanh()

    def init_weights(self, init_w):
        self.dense_1.weight.data = fanin_init(self.dense_1.weight.data.size())
        self.dense_2.weight.data = fanin_init(self.dense_2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

    def forward(self, input):
        x = self.dense_1(input)
        x = self.relu1(x)
        x = self.dense_2(x)
        x = self.relu2(x)
        output = self.output(x)
        output = self.tanh(output)
        return output


class CriticDDPGNetwork(nn.Module):

    # The Critic Network basically takes the state and action as the input and outputs a q value
    def __init__(self, num_conv_layers, conv_kernel_size, input_channels, output_q_value, dense_layer,
                pool_kernel_size, IMG_HEIGHT, IMG_WIDTH):
        super(CriticDDPGNetwork, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels
        self.output_q_value = output_q_value
        self.dense_layer = dense_layer
        self.pool_kernel_size = pool_kernel_size
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH

        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.pool_kernel_size)
        self.bn1 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.pool_kernel_size)
        self.bn2 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu2 = nn.ReLU(inplace=True)
        self.fully_connected_layer = nn.Linear(234432, self.dense_layer)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.dense_layer, output_q_value)

        # Weight initialization from a uniform gaussian distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, states, actions):
        x = self.conv1(states)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        x = x + actions # Adding the action input
        x = self.relu3(x)
        output = self.output(x)
        return output


class CriticDDPGNonConvNetwork(nn.Module):

    def __init__(self, num_hidden_layers, output_q_value, input,
                 action_dim, goal_dim):
        super(CriticDDPGNonConvNetwork, self).__init__()
        # Initialize the variables
        self.num_hidden = num_hidden_layers
        self.output_dim = output_q_value
        self.input = input
        self.action_dim = action_dim
        self.init_w = 3e-3

        # Dense Block
        self.dense1 = nn.Linear(self.input, self.num_hidden)
        self.relu1 = nn.ReLU(inplace=True)
        self.hidden2 = nn.Linear(self.num_hidden + self.action_dim, self.num_hidden)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.num_hidden, self.output_dim)

    def init_weights(self, init_w):
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.hidden2.weight.data = fanin_init(self.hidden2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

    def forward(self, states, actions):
        #print(states)
        x = self.dense1(states)
        x = self.relu1(x)
        x = torch.cat((x, actions), dim=1)
        x = self.hidden2(x)
        x = self.relu3(x)
        out = self.output(x)
        return out