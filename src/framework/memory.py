from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))


class Memory(object):
    def __init__(self):
        self.memory = deque()

    def push(self, state, action, reward, policy):
        self.memory.append(Transition(state, action, reward, policy))

    def sample(self):
        memory = self.memory
        return Transition(*zip(*memory))

    def __len__(self):
        return len(self.memory)
