from RL import Agent
from utils import plotLearning
import numpy as np
import gym
import tensorflow as tf 

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH "]='True'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    lr = 0.001
    n_games = 1
    agent = Agent(gamma=0.99, epsilon=0, lr=lr,
                  input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n, mem_size=1000000,
                  batch_size=64, epsilon_end=0.01,epsilon_dec=1e-4)
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
    
        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score, 'average_score %.2f' % avg_score, 
            'epsilon %.2f' % agent.epsilon)
    
    filename = 'lunarlander_tf2.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)