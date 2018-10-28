# -*- coding: utf-8 -*-
import random
# import gym
import numpy as np
import tensorflow as tf
from collections import deque

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, Flatten, Lambda, merge, concatenate
from keras.optimizers import Adam,SGD, RMSprop
from keras import backend as K
from keras import regularizers

import plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# EPISODES = 5000

from prioritized_memory import Memory


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99    # discount rate
        self.no_op_steps = 30

        # optimizer parameters
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.threads = 7

        # create model for actor and critic network
        self.actor, self.critic = self._build_model()
        self.local_actor, self.local_critic = self.build_localmodel()
        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.episode = 1
        self.__print_period = 100
        self.__save_graph_period = 100
        self.__episode_list = []
        self.__score_list = []
        self.__step_list = []
        self.__Max_reward = []
        self.t_max = 20
        self.t = 0
        self.states = []
        self.rewards = []
        self.actions = []

        self.avg_p_max = 0

    #정책 신경망과 가치 신경망을 생성
    def _build_model(self):
        input = Input(shape=(self.state_size,))

        dense1 = Dense(24, activation='relu')(input)
        dense2 = Dense(256, activation='relu')(dense1)
        dense3 = Dense(256, activation='relu')(dense2)
        fc = Dense(256, activation='relu')(dense3)

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        #가치와 정책을 예측하는 함수를 만든다
        actor._make_predict_function()
        critic._make_predict_function()

        #actor.summary()
        #critic.summary()

        return actor, critic
    # make agents(local) and start training
    def train(self):
        # self.load_model('./save_model/cartpole_a3c.h5')
        agents = [Agent(i, self.actor, self.critic, self.optimizer, self.discount_factor,
                        self.action_size, self.state_size) for i in range(self.threads)]

        for agent in agents:
            agent.start()

    #정책신경망을 업데이트 하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = actor_loss + 0.01*entropy
        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + '_critic.h5')

    def save_learning_result(self, score, maxnumber, Max_prob):
        self.episode += 1
        # self.__global_step += step
        if (self.episode % self.__print_period == 0):
            print("episode:{0:>8}".format(self.episode), " score:{0:>5}".format(score),
                  "MaxNumber:{0:9}".format(maxnumber), " step:{0:>5}".format(Max_prob))
            # " avg actor loss:{0:>6.4f}".format(avg_actor_loss),
            # " avg critic loss:{0:>6.4f}".format(avg_critic_loss))
        # 정보를 저장한다.
        self.__episode_list.append(self.episode)
        self.__score_list.append(score)
        self.__step_list.append(maxnumber)
        self.__Max_reward.append(Max_prob)
        # self.__avg_max_p_list.append(Max_Q_value)
        # self.__avg_actor_loss_list.append(avg_actor_loss)
        # self.__avg_critic_loss_list.append(avg_critic_loss)
        # 그래프를 그린다.
        if (self.episode % self.__save_graph_period == 0):
            trace1 = go.Scatter(x=self.__episode_list, y=self.__score_list, name='score')
            trace2 = go.Scatter(x=self.__episode_list, y=self.__step_list, name='MaxNumber')
            trace3 = go.Scatter(x=self.__episode_list, y=self.__Max_reward, name='Max_prob')
            # trace4 = go.Scatter(x=self.__episode_list, y=self.__avg_max_p_list, name='Max_Q_value')
            # trace5 = go.Scatter(x=self.__episode_list, y=self.__avg_actor_loss_list, name='average actor loss')
            # trace6 = go.Scatter(x=self.__episode_list, y=self.__avg_critic_loss_list, name='average critic loss')

            fig = tools.make_subplots(rows=3, cols=2, print_grid=False)
            fig.append_trace(trace1, 1, 1)
            fig.append_trace(trace2, 1, 2)
            fig.append_trace(trace3, 2, 1)
            # fig.append_trace(trace4, 2, 2)
            # fig.append_trace(trace5, 3, 1)
            # fig.append_trace(trace6, 3, 2)

            fig['layout']['xaxis1'].update(title='episode')
            fig['layout']['xaxis2'].update(title='episode')
            fig['layout']['xaxis3'].update(title='episode')
            # fig['layout']['xaxis4'].update(title='episode')
            # fig['layout']['xaxis5'].update(title='episode')
            # fig['layout']['xaxis6'].update(title='episode')

            fig['layout']['yaxis1'].update(title='score')
            fig['layout']['yaxis2'].update(title='MaxNumber')
            fig['layout']['yaxis3'].update(title='Max_prob')
            # fig['layout']['yaxis4'].update(title='average max prob')
            # fig['layout']['yaxis5'].update(title='average actor loss', range=[-1, 1])
            # fig['layout']['yaxis6'].update(title='average critic loss')

            fig['layout'].update(title='Learning Result')
            py.offline.plot(fig, "learning.html", auto_open=False)

    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([self.states, self.actions, advantages])
        self.optimizer[1]([self.states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policy = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def build_localmodel(self):
        input = Input(shape=(self.state_size,))

        dense1 = Dense(24, activation='relu')(input)
        dense2 = Dense(256, activation='relu')(dense1)
        dense3 = Dense(256, activation='relu')(dense2)
        fc = Dense(256, activation='relu')(dense3)

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만든다
        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()
        return local_actor, local_critic

    def update_localmodel(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())