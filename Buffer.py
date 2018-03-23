from collections import  namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'achieved_goal', 'desired_goal'
                         ,'new_achieved_goal', 'new_desired_goal', 'success'))

class ReplayBuffer(object):

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