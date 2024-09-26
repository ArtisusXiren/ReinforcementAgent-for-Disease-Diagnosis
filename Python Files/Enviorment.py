#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from gym import spaces
import numpy as np
from numpy import random

# In[2]:


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv,self).__init__()
        self.observation_space=spaces.Dict({
            "temperature":spaces.Box(low=95.8,high=105.0,shape=(1,),dtype=np.float32),
            "symptoms": spaces.MultiBinary(5),
            "age_group": spaces.Discrete(4) })
        self.action_space=spaces.Dict({
            "treatment":spaces.Discrete(5),
            "dosage":spaces.Box(low=0.0,high=100.0,shape=(1,),dtype=np.float32)})
        self.state={
            "temperature":np.array([98.6],dtype=np.float32),
            "symptoms": np.array([0,0,0,0,0],dtype=int),
            "age_group":2}
        self.done=False
        self.symptoms_to_flu=np.array([1,1,1,0,0])
        self.correct_diagnosis_reward=10
        self.incorrect_diagnosis_penalty=-10
        self.asking_penalty=-1
    def reset(self):
        self.state={
            "temperature":np.array([98.6],dtype=np.float32),
            "symptoms": random.choice([0,1],size=(5,)),
            "age_group":random.choice(4)
        }
        self.done=False
        return tuple(self.state.values())
    def step(self,action):
        treatment =action['treatment']
        if treatment<5:
            reward=self.asking_penalty
            self.state['symptoms'][treatment]=self.symptoms_to_flu[treatment]
            if np.all(self.state['symptoms'] == self.symptoms_to_flu):
                reward = self.correct_diagnosis_reward
                self.done = True
            else:
                reward = self.asking_penalty
        elif treatment == 5:
        # Check if symptoms match symptoms_to_flu array
            if np.all(self.state['symptoms'] == self.symptoms_to_flu):
                reward = self.correct_diagnosis_reward
                self.done = True
        elif treatment == 6:
        # Check if symptoms do not match symptoms_to_flu array
            if np.all(self.state['symptoms'] != self.symptoms_to_flu):
                reward = self.correct_diagnosis_reward
                self.done = True
            else:
                reward = self.incorrect_diagnosis_penalty
                self.done = True
        return tuple(self.state.values()),reward,self.done,{}
    def render(self,mode='human'):
        print(f"State:{self.state}")
    def close(self):
        pass
    
        
            
            


# In[ ]:




