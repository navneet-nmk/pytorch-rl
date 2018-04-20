from collections import  namedtuple, deque
from torch.utils.data import Dataset
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done', 'success'))


class ReplayBufferDataset(Dataset):
    """
    Dataset implementation of the experience replay
    This class helps in the case of using multi gpu training
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]


class ReplayBuffer(object):

    def __init__(self, capacity, seed, priotirized_experience=False):
        self.capacity = capacity
        self.position = 0
        self.prioritize = priotirized_experience
        if self.prioritize:
            self.memory = []
        else:
            self.memory = []
        # Seed for reproducible results
        np.random.seed(seed)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)


# Use this replay buffer for non goal environments
class ReplayBufferDeque(object):

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


# Long term memory that uses resevoir sampling for adding items
class SelectiveExperienceReplayBuffer(object):
    pass







