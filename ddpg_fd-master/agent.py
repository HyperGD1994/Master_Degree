import abc

class Agent(object):

    @abc.abstractmethod
    def initialize_episode(self, episode_count):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, observation):
        raise NotImplementedError

    @abc.abstractmethod
    def feedback(self, resulting_state, reward, episode_done):
        raise NotImplementedError

    @abc.abstractmethod
    def set_learning(self, learning_flag):
        raise NotImplementedError
