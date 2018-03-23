import tensorflow as tf
import numpy as np
import gym
import time

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODE = 200
MAX_EP_STEP = 200
LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9

MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True

ENV_NAME = 'Pendulum-v0'

REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement

        self.t_replace_counter = 0

        with tf.name_scope('Actor'):
            # input s, out put a
            self.a = self._build_net(S, scope='eval_net', trainable=True)
            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1-self.replacement['tau'])*t+self.replacement['tau']*e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.name_scope(scope):
            init_w = tf.random_normal_initializer(mean=0., stddev=0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(
                inputs=s,
                units=30,
                activation=tf.nn.relu,
                kernel_initializer=init_w,
                bias_initializer=init_b,
                name='l1',
                trainable=trainable
            )

            with tf.name_scope('a'):
                actions = tf.layers.dense(
                    inputs=net,
                    units=self.a_dim,
                    activation=tf.nn.tanh,
                    kernel_initializer=init_w,
                    bias_initializer=init_b,
                    name='a',
                    trainable=trainable
                )
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')

        return scaled_a

    def learn(self, s):
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action = self.sess.run(self.a, feed_dict={S: s})

        return action[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr) # -lr for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        pass

    def _build_net(self):
        pass

    def learn(self, s, a, r, s_):
        pass


class Memory(object):
    def __init__(self, capactiy, dims):
        pass

    def store_transition(self, s, a, r, s_):
        pass

    def sample(self, n):
        pass


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')

with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, shape=[None, 1], name='r')

with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
