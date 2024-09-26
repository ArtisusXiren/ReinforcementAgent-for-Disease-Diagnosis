#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import gym
from gym import spaces
from collections import defaultdict
from typing import Tuple
from Enviorment import CustomEnv


# In[2]:


class learningAgent:
    def __init__(self,action_space,learning_rate=0.1,discount_factor=0.99,exploration_rate=1.0,exploration_decay=0.995):
        self.action_space=action_space
        self.lr=learning_rate
        self.df=discount_factor
        self.er=exploration_rate
        self.ed=exploration_decay
        self.q_table = defaultdict(lambda: np.zeros(action_space.spaces['treatment'].n))

        
    def chose_action(self,state):
        if random.uniform(0,1)<self.er:
            treatment_action = self.action_space['treatment'].sample()
            dosage_action = float(self.action_space['dosage'].sample())
            return {"treatment": treatment_action, "dosage": dosage_action}
        else:
            state = tuple(map(lambda x: tuple(x) if isinstance(x, np.ndarray) else x, state))
            treatment_action = np.argmax(self.q_table[state])
            return {"treatment": treatment_action, "dosage": 0.0}
    def learn(self,state,action,reward,next_state,done):
        state = tuple(map(lambda x: tuple(x) if isinstance(x, np.ndarray) else x, state))
        next_state = tuple(map(lambda x: tuple(x) if isinstance(x, np.ndarray) else x, next_state))
        best_next_action=np.argmax(self.q_table[next_state])
        temporal_difference=reward+self.df*self.q_table[next_state][best_next_action]*(1-done)
        td_delta=temporal_difference-self.q_table[state][action['treatment']]
        self.q_table[state][action['treatment']]+=self.lr*td_delta
        if done:
            self.er*=self.ed


# In[6]:


env=CustomEnv()
agent=learningAgent(env.action_space)
num_episodes=1000
for episodes in range (num_episodes):
    state=env.reset()
    done=False
    total_reward=0
    while not done:
        action=agent.chose_action(state)
        next_state, reward, done, info=env.step(action)
        agent.learn(state,action,reward,next_state,done)
        state=next_state
        total_reward+=reward
    print(f"Episode {episodes+1}/{num_episodes}-Total Reward:{total_reward}")
env.close()
        


# In[ ]:





# In[ ]:




