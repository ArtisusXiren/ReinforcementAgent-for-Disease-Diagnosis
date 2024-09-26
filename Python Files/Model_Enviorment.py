import gymnasium as gym
import numpy as np 
from gymnasium import spaces
from Influenza_model import X_test,y_test
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import joblib
class ModelEnv(gym.Env):
    def __init__(self):
        super(ModelEnv,self).__init__()
        self.model=joblib.load('Model.pkl')
        self.prize=10
        self.penality=-1
        self.action_space=spaces.Discrete(3)
        self.observation_space=spaces.Box(low=0,high=1,shape=(12,),dtype=np.int32)
        self.state=np.random.randint(2,size=(12,))
    def reset(self,seed=None,options=None):
        self.state=np.random.randint(2,size=(12,))
        self.current_Step=0
        return self.state,{}
    def step(self,action):
        self.state[action]=1-self.state[action]
        prediction=self.model.predict([self.state])
        true_label,similarity=self.get_true_label(X_test,y_test)
        accuracy=accuracy_score([true_label],prediction)
        self.current_Step+=1
        max_steps=200
        if accuracy>0.85:
            reward=float(self.prize*accuracy+similarity)
        elif 0.65<=accuracy<=0.85:
            reward = float((self.prize*accuracy)/2 +similarity/2)
        else:
            reward= float(self.penality/2 + similarity/4)
        terminated = True if (np.all(reward>=self.prize)) or (self.current_Step>= max_steps) else False
        truncated=False
        return self.state,reward,terminated,truncated,{}   
    def get_true_label(self,X_test,y_test):
       similarity=cosine_similarity([self.state],X_test)[0].max()
       test_index=similarity.argmax()
       true_label=y_test[test_index]
       return true_label,similarity    
    def render(self,reward,mode='human'):
        print(f'State:{self.state},reward:{reward}')
    def close():
        pass        
       