
import math
import numpy as np
import os


class TransitionChunk(object):

    def __init__(self, states, actions, rewards, next_states, is_terminal):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.is_terminal = is_terminal


class ActionChunk(object):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.weights = None


class Memory(object):

    def __init__(self, max_capacity, state_shape):
        self._max_capacity = max_capacity
        self._insert_index = 0
        self._num_entries = 0

        state_shape = (max_capacity, ) + state_shape

        self._states = np.zeros(state_shape)
        self._actions = np.zeros(max_capacity)
        self._rewards = np.zeros(max_capacity)
        self._next_states = np.zeros(state_shape)
        self._is_terminal = np.zeros(max_capacity, dtype=np.bool)

    def initialize_episode(self, episode_count):
        pass

    def num_entries(self):
        return self._num_entries

    def capacity(self):
        return self._max_capacity

    def add_memory(self, state, action, reward, next_state):
        if self._insert_index >= self._max_capacity:
            self._insert_index = 0

        self._states[self._insert_index] = state
        self._actions[self._insert_index] = action
        self._rewards[self._insert_index] = reward

        if next_state is None:
            self._is_terminal[self._insert_index] = True
        else:
            self._is_terminal[self._insert_index] = False
            self._next_states[self._insert_index] = next_state

        self._insert_index += 1

        if self._num_entries < self._max_capacity:
            self._num_entries += 1

    def sample_transitions(self, num_samples):
        indices = np.random.choice(np.arange(self._num_entries),
                                   size=num_samples)

        return TransitionChunk(self._states[indices],
                               self._actions[indices],
                               self._rewards[indices],
                               self._next_states[indices],
                               self._is_terminal[indices])

    def sample_actions(self, num_samples):
        indices = np.random.choice(np.arange(self._num_entries),
                                   size=num_samples)

        return ActionChunk(self._states[indices],
                           self._actions[indices])


def from_demonstrations(dir_path, state_shape):
    files = os.listdir(dir_path)

    observations = []
    actions = []
    rewards = []

    for fn in files:
        path = os.path.join(dir_path, fn)
        file = open(path, 'r')

        observations.append(np.squeeze(np.load(file)))
        actions.append(np.squeeze(np.load(file)))
        rewards.append(np.squeeze(np.load(file)))

        file.close()

    total_entries = 0
    for obs in observations:
        total_entries += obs.shape[0]

    demo_memory = Memory(total_entries, state_shape)

    for obs, acts, rs in zip(observations, actions, rewards):
        assert obs.shape[0] == acts.shape[0]
        assert obs.shape[0] == rs.shape[0]

        for i in range(obs.shape[0]):
            state = obs[i]
            action = acts[i]
            reward = rs[i]

            next_state = obs[i + 1] if i < (obs.shape[0] - 1) else None
            demo_memory.add_memory(state, action, reward, next_state)

    return demo_memory

