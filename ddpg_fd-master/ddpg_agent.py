
import agent
import memory
import actor_network
import critic_network

import math
import random
import numpy as np
import tensorflow as tf

from gym.spaces.discrete import Discrete

_ACTOR_LAYERS = (16, 16)
_ACTOR_ACTIVATION = tf.nn.relu

_CRITIC_LAYERS = (16, 16)
_CRITIC_LAYER_ACTIVATION = tf.nn.relu
_CRITIC_OUTPUT_ACTIVATION = tf.identity

_TARGET_UPDATE_RATE = 1000
_LEARN_BATCH_SIZE = 256
_DISCOUNT = 0.98


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class DDPGAgent(agent.Agent):

    def __init__(self, action_space, observation_space, exploration_rate, replay_buffer,
                 positive_demos, negative_demos):
        self._action_space = action_space
        self._observation_space = observation_space
        self._exploration_rate = exploration_rate
        self._actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self._action_space.shape[0]))

        self._state_shape = observation_space.high.shape
        self._memory = replay_buffer
        self._positive_demos = positive_demos
        self._negative_demos = negative_demos

        self._cur_exploration = self._exploration_rate(0)

        self._last_action = None
        self._last_state = None

        self._build_graph()

        self._sess = tf.Session(graph=self._graph)
        with self._sess.as_default():
            self._sess.run(self._init_op)
            self._sess.run(self._target_copy_ops)

    def pretrain_actor(self, iters):
        for _ in range(iters):
            # demo_chunk = self._sample_demo_actions_chunk()
            demo_chunk = self._positive_demos.sample_actions(_LEARN_BATCH_SIZE)
            feed_dict = {
                    self._state : demo_chunk.states,
                    self._expert_action : demo_chunk.actions.reshape(-1, self._action_space.shape[0]),
            }

            with self._sess.as_default():
                self._sess.run((self._actor_pretrain_optimizer), feed_dict=feed_dict)

        with self._sess.as_default():
            self._sess.run(self._target_copy_ops)

    def _sample_demo_actions_chunk(self):
        num_positive = _LEARN_BATCH_SIZE#  * 3 / 4
        num_negative = _LEARN_BATCH_SIZE - num_positive

        positive_demos = self._positive_demos.sample_actions(num_positive)
        negative_demos = self._negative_demos.sample_actions(num_negative)

        states = np.concatenate([positive_demos.states, negative_demos.states], axis=0)
        actions = np.concatenate([positive_demos.actions, negative_demos.actions], axis=0)
        result_chunk = memory.ActionChunk(states, actions)
        result_chunk.weights = np.array(([1.0] * num_positive) + ([-0.1] * num_negative))

        return result_chunk

    def _sample_replay_buffer(self):
        return self._memory.sample_transitions(_LEARN_BATCH_SIZE)

        num_replay = _LEARN_BATCH_SIZE# * 2 / 3
        num_demos = _LEARN_BATCH_SIZE - num_replay

        replay = self._memory.sample_transitions(num_replay)
        positive_demos = self._positive_demos.sample_transitions(num_demos)

        states = np.concatenate([replay.states, positive_demos.states], axis=0)
        actions = np.concatenate([replay.actions, positive_demos.actions], axis=0)
        rewards = np.concatenate([replay.rewards, positive_demos.rewards], axis=0)
        next_states = np.concatenate([replay.next_states, positive_demos.next_states], axis=0)
        is_terminal = np.concatenate([replay.is_terminal, positive_demos.is_terminal], axis=0)

        return memory.TransitionChunk(states, actions, rewards, next_states, is_terminal)


    def pretrain_critic(self, iters):
        for _ in range(iters):
            mem_chunk = self._positive_demos.sample_transitions(_LEARN_BATCH_SIZE)
            feed_dict = {
                    self._state : mem_chunk.states,
                    self._action : mem_chunk.actions.reshape(-1, self._action_space.shape[0]),
                    self._reward : mem_chunk.rewards,
                    self._next_state : mem_chunk.next_states,
                    self._target_is_terminal : mem_chunk.is_terminal,
            }

            with self._sess.as_default():
                self._sess.run((self._critic_optimizer, self._critic_margin_optimizer),
                               feed_dict=feed_dict)
                self._sess.run(self._target_update_ops)

        with self._sess.as_default():
            self._sess.run(self._target_copy_ops)

    def initialize_episode(self, episode_count):
        self._cur_exploration = self._exploration_rate(episode_count)
        self._memory.initialize_episode(episode_count)

    def act(self, observation):
        observation = self._normalised_state(observation)

        self._learn()

        observation = observation.reshape((1,) + self._state_shape)
        feed_dict = {
            self._act_noise: self._cur_exploration,
            self._act_observation: observation,
        }

        with self._sess.as_default():
            action = self._sess.run(self._act_output, feed_dict=feed_dict)

        self._last_state = observation
        self._last_action = action

        return action + self._actor_noise()

    def feedback(self, resulting_state, reward, episode_done):
        resulting_state = resulting_state.reshape((1,) + self._state_shape)
        resulting_state = self._normalised_state(resulting_state)

        if episode_done:
            resulting_state = None

        self._memory.add_memory(self._last_state, self._last_action, reward, resulting_state)

    def set_learning(self, learning_flag):
        self._learning_flag = learning_flag

    def _learn(self):
        if self._memory.num_entries() < 1000: #self._memory.capacity() / 10:
            return

        mem_chunk = self._sample_replay_buffer() #self._memory.sample_transitions(_LEARN_BATCH_SIZE)
        feed_dict = {
                self._state : mem_chunk.states,
                self._action : mem_chunk.actions.reshape(-1, self._action_space.shape[0]),
                self._reward : mem_chunk.rewards,
                self._next_state : mem_chunk.next_states,
                self._target_is_terminal : mem_chunk.is_terminal,
        }

        with self._sess.as_default():
            self._sess.run((self._critic_optimizer), feed_dict=feed_dict)
            self._sess.run((self._actor_optimizer), feed_dict=feed_dict)

            self._sess.run(self._target_update_ops)

    def _build_graph(self):
        self._graph = tf.Graph()

        action_ranges = (self._action_space.low, self._action_space.high)

        with self._graph.as_default():
            self._actor = actor_network.ActorNetwork(_ACTOR_LAYERS,
                                                     action_ranges,
                                                     _ACTOR_ACTIVATION)
            self._target_actor = actor_network.ActorNetwork(_ACTOR_LAYERS,
                                                            action_ranges,
                                                            _ACTOR_ACTIVATION)

            self._critic = critic_network.CriticNetwork(_CRITIC_LAYERS,
                                                        _CRITIC_LAYER_ACTIVATION,
                                                        _CRITIC_OUTPUT_ACTIVATION)
            self._target_critic = critic_network.CriticNetwork(_CRITIC_LAYERS,
                                                               _CRITIC_LAYER_ACTIVATION,
                                                               _CRITIC_OUTPUT_ACTIVATION)

            self._state = tf.placeholder(
                    tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))
            self._action =  tf.placeholder(
                    tf.float32, shape=(_LEARN_BATCH_SIZE, self._action_space.shape[0]))
            self._expert_action =  tf.placeholder(
                    tf.float32, shape=(_LEARN_BATCH_SIZE, self._action_space.shape[0]))
            self._reward = tf.placeholder(tf.float32, shape=_LEARN_BATCH_SIZE)
            self._next_state = tf.placeholder(
                    tf.float32, shape=((_LEARN_BATCH_SIZE, ) + self._state_shape))
            self._target_is_terminal = tf.placeholder(tf.bool, shape=_LEARN_BATCH_SIZE)

            self._build_acting_network()

            self._build_actor_learning_network()
            self._build_critic_learning_network()

            self._build_actor_pretrain_network()
            self._build_critic_pretrain_network()

            self._build_copy_ops()
            self._build_update_ops()

            self._init_op = tf.global_variables_initializer()

    def _build_acting_network(self):
        self._act_observation = tf.placeholder(tf.float32, shape=((1, ) + self._state_shape))
        self._act_noise = tf.placeholder(tf.float32)
        self._act_output = self._actor(self._act_observation)
                            # tf.random_normal(shape=(1,), stddev=self._act_noise))

    def _build_actor_learning_network(self):
        action = self._actor(self._state)

        # TODO: should this be the critic or target_critic?
        qvalue = self._critic(self._state, action)
        self._actor_loss = -qvalue # * self._normalized_weights

        opt = tf.train.AdamOptimizer(0.0001)
        self._actor_optimizer = opt.minimize(self._actor_loss,
                                             var_list=self._actor.get_variables())

    def _build_actor_pretrain_network(self):
        stddev = tf.constant([0.01, 0.01])
        offset = tf.random_normal(self._state.get_shape(), stddev=stddev)
        action = self._actor(self._state + offset)

        pretrain_loss = tf.losses.mean_squared_error(action, self._expert_action)

        opt = tf.train.AdamOptimizer()
        self._actor_pretrain_optimizer = opt.minimize(
            pretrain_loss, var_list=self._actor.get_variables())

    def _build_critic_learning_network(self):
        critic_output = tf.reshape(self._critic(self._state, self._action), [-1])
        self._critic_output = critic_output

        next_state_action = self._target_actor(self._next_state)
        next_state_qvalue = tf.reshape(self._target_critic(self._next_state, next_state_action), [-1])

        terminating_target = self._reward
        intermediate_target = self._reward + next_state_qvalue * _DISCOUNT
        desired_output = tf.where(self._target_is_terminal, terminating_target, intermediate_target)

        self._critic_loss = tf.losses.mean_squared_error(desired_output, critic_output)

        opt = tf.train.AdamOptimizer(0.0001)
        self._critic_optimizer = opt.minimize(self._critic_loss,
                                              var_list=self._critic.get_variables())

    def _build_critic_pretrain_network(self):
        stddev = tf.constant([0.1, 0.01])
        offset = tf.random_normal(self._state.get_shape(), stddev=stddev)
        state = self._state# + offset

        demo_action_q = tf.stop_gradient(tf.reshape(self._critic(state, self._action), [-1]))

        random_action = tf.random_uniform(shape=(_LEARN_BATCH_SIZE, self._action_space.shape[0]),
                                          minval=-1.0, maxval=1.0)
        random_action_q = tf.reshape(self._critic(state, random_action), [-1])

        margin = 1.0
        margin_loss = tf.square(tf.maximum(0., (random_action_q + margin) - demo_action_q))

        margin_loss = tf.reduce_mean(margin_loss)

        opt = tf.train.AdamOptimizer()
        self._critic_margin_optimizer = opt.minimize(
            margin_loss, var_list=self._critic.get_variables())

    def _build_update_ops(self):
        actor_vars = self._actor.get_variables()
        actor_target_vars = self._target_actor.get_variables()
        tau = 0.01

        self._target_update_ops = []

        for src_var, dst_var in zip(actor_vars, actor_target_vars):
            self._target_update_ops.append(
                dst_var.assign(tf.multiply(src_var, tau) + tf.multiply(dst_var, 1.0 - tau)))

        # for src_var, dst_var in zip(actor_vars, actor_target_vars):
        #     self._target_update_ops.append(
        #             tf.assign(dst_var, src_var, validate_shape=True, use_locking=True))

        critic_vars = self._critic.get_variables()
        critic_target_vars = self._target_critic.get_variables()

        for src_var, dst_var in zip(critic_vars, critic_target_vars):
            self._target_update_ops.append(
                dst_var.assign(tf.multiply(src_var, tau) + tf.multiply(dst_var, 1.0 - tau)))

        # for src_var, dst_var in zip(critic_vars, critic_target_vars):
        #     self._target_update_ops.append(
        #             tf.assign(dst_var, src_var, validate_shape=True, use_locking=True))

    def _build_copy_ops(self):
        actor_vars = self._actor.get_variables()
        actor_target_vars = self._target_actor.get_variables()

        self._target_copy_ops = []

        for src_var, dst_var in zip(actor_vars, actor_target_vars):
            self._target_copy_ops.append(
                    tf.assign(dst_var, src_var, validate_shape=True, use_locking=True))

        critic_vars = self._critic.get_variables()
        critic_target_vars = self._target_critic.get_variables()

        for src_var, dst_var in zip(critic_vars, critic_target_vars):
            self._target_copy_ops.append(
                    tf.assign(dst_var, src_var, validate_shape=True, use_locking=True))

    def _normalised_state(self, obs):
        # obs[0] /= self._observation_space.high[0] / 2.0
        # obs[1] /= 1.5
        # obs[2] /= self._observation_space.high[2] / 2.0
        # obs[3] /= 1.5
        return obs
        # obs_range = (self._observation_space.high - self._observation_space.low)
        # return (obs - self._observation_space.low) / obs_range
