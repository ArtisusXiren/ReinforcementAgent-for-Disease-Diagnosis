import gymnasium as gym
import pydicom
from PIL import Image
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DLEnv(gym.Env):
    def __init__(self,val_images,val_labels):
        super(DLEnv,self).__init__()
        self.val_images=val_images
        self.val_labels=val_labels
        self.prize=10
        self.penality=-1
        self.previous_reward=0
        self.learning_rate_options = [0.0001, 0.001, 0.005, 0.01]
        self.dropout_rate_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.action_space=spaces.MultiDiscrete((
            len(self.learning_rate_options),
            len(self.dropout_rate_options),
            3,
            4,
            2
            
        ))
        self.observation_space=spaces.Box(low=0,high=255,shape=(128,128,1),dtype=np.uint8)
        self.current_idx=0
        self.max_steps=len(val_images)
        
    def make_prediction(self,img):
         prediction=self.model.predict(img)
         return prediction  
        
    def load_image(self,dcm_path):
        print(f"[DEBUG] Loading image: {dcm_path}")
        img = load_img(dcm_path, target_size=(128, 128), color_mode="grayscale")
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  
        return img
    
    def step(self,action):
        print(f"[DEBUG] Step called with action: {action} (current_idx={self.current_idx})")
        self.model=load_model(r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\model.keras')
        learning_rate=self.learning_rate_options[action[0]]
        dropout_rate=self.dropout_rate_options[action[1]]
        batch_size=[8,16,32][action[2]]
        augmentation=action[3]
        normlization=action[4]
        
        augmentation_options ={
            0:{'horizontal_flip': True},
            1:{'rotation_range':20},
            2:{'zoom_range':0.2},
            3:{}
        }
        selected_augumentation=augmentation_options.get(augmentation,{})
        
        normlization_options={
            0:{'featurewise_center': True, 'featurewise_std_normalization':True},
            1:{'rescale':1./255},
            2:{}
        }
        
        selected_normalization=normlization_options.get(normlization,{})
        test_image=self.load_image(self.val_images[self.current_idx])    
        datagen=ImageDataGenerator(**selected_augumentation,**selected_normalization)
        
        pred_img=datagen.random_transform(test_image[0])
        pred_img = np.expand_dims(pred_img, axis=0)
        print("[DEBUG] Making prediction...")
        prediction=self.make_prediction(pred_img)
        predicted_label=np.argmax(prediction)
        
        true_label=self.val_labels[self.current_idx] 
        
        if predicted_label==true_label:
            reward=self.prize
            self.previous_reward=self.prize
        else:
            reward=self.previous_reward + self.penality
        print(f"[DEBUG] Prediction: {predicted_label}, True Label: {true_label}, Reward: {reward}")
        
        #next_image=self.load_image(self.val_images[self.current_idx])
        self.truncated=False
        done = self.current_idx >= len(self.val_images) - 1

        if done:
            print("[DEBUG] Reached last image. Setting done=True")
            next_image = np.zeros_like(pred_img)  
        else:
            self.current_idx+=1
            next_image = self.load_image(self.val_images[self.current_idx])
            
        print(f"[DEBUG] Step completed. Next Index: {self.current_idx}, Done: {done}")
        return next_image,reward,done,self.truncated,{}
    
    def reset(self,seed=None):
        self.current_idx=0
        self.done = False 
        self.current_image = self.load_image(self.val_images[self.current_idx])   
        return self.current_image ,{}
            
    def render(self,mode='human'):
        pass 
        
val_images=[r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\0\ID00007637202177411956430_1.dcm.jpg',r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\1\ID00012637202177665765362_1.dcm.jpg',r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\1\ID00426637202313170790466_403.dcm.jpg',r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\2\ID00082637202201836229724_3.dcm.jpg',r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\2\ID00355637202295106567614_7.dcm.jpg']
val_labels=[0,1,1,2,2]
env=DLEnv(val_images,val_labels)
#vec_env=make_vec_env(lambda:env,n_envs=1)

model=PPO("CnnPolicy", env,verbose=1)
model.learn(total_timesteps=1000)
model.save("RL_agent")

model = PPO.load("RL_agent")
env= DLEnv(val_images,val_labels)
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    print(f"[DEBUG] Step: Reward {reward}, Done: {done}")
    
    if done: 
        break
print("[DEBUG] All steps completed. Stopping loop.")