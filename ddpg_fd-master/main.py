
import agent
import gym
import sys
import time

import run_loop
import decaying_value
import ddpg_agent
# import keyboard_agent
import memory
import observer


TRAIN_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 1000


def _build_observers(env):
    observers = []
    observers.append(observer.RewardTracker())
    # observers.append(observer.SavingObserver('negative_demos/'))
    # observers.append(observer.Renderer(env, 20.))
    return observers


# env = gym.make('Pendulum-v0')
env = gym.make('MountainCarContinuous-v0')
# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('CarRacing-v0')
env.reset()

print('action space: {}'.format(env.action_space))
print('act low/high: {} {}'.format(env.action_space.low, env.action_space.high))
print('obs low/high: {} {}'.format(env.observation_space.low, env.observation_space.high))

observers = _build_observers(env)
exploration_rate = decaying_value.DecayingValue(0.2, 0.1, TRAIN_EPISODES)

replay_buffer = memory.Memory(100000, env.observation_space.high.shape)
positive_demos = memory.from_demonstrations('positive_demos/', env.observation_space.high.shape)
# negative_demos = memory.from_demonstrations('negative_demos/', env.observation_space.high.shape)

agent = ddpg_agent.DDPGAgent(env.action_space, env.observation_space, exploration_rate,
                             replay_buffer, positive_demos, None)

agent.pretrain_actor(2000)
agent.pretrain_critic(2000)


run_loop.run_loop(env, agent, TRAIN_EPISODES, MAX_STEPS_PER_EPISODE, observers)
wait = raw_input("Finished Training")

agent.set_learning(False)

observers.append(observer.Renderer(env, 20.))
# agent = keyboard_agent.KeyboardAgent(env)
run_loop.run_loop(env, agent, 10, None, observers)
