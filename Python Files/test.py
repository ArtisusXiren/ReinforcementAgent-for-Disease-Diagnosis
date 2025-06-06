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
    def __init__(self, val_images, val_labels):
        super(DLEnv, self).__init__()
        self.val_images = val_images
        self.val_labels = val_labels
        self.prize = 10
        self.penality = -1
        self.learning_rate_options = [0.0001, 0.001, 0.005, 0.01]
        self.dropout_rate_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.action_space = spaces.MultiDiscrete((
            len(self.learning_rate_options),
            len(self.dropout_rate_options),
            3,
            4,
            2
        ))
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8)
        self.current_idx = 0
        self.max_steps = len(val_images)  # Stop after all images are processed
        self.model = load_model(r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\model.keras')  # Load once

    def load_image(self, dcm_path):
        print(f"[DEBUG] Loading image: {dcm_path}")
        img = load_img(dcm_path, target_size=(128, 128), color_mode="grayscale")
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def step(self, action):
        print(f"[DEBUG] Step called with action: {action} (current_idx={self.current_idx})")

        if self.current_idx >= self.max_steps:
            print("[ERROR] self.current_idx exceeded max_steps! Something went wrong.")
            return np.zeros((128, 128, 1), dtype=np.uint8), 0, True, False, {}

        learning_rate = self.learning_rate_options[action[0]]
        dropout_rate = self.dropout_rate_options[action[1]]
        batch_size = [8, 16, 32][action[2]]
        augmentation = action[3]
        normlization = action[4]

        augmentation_options = {
            0: {'horizontal_flip': True},
            1: {'rotation_range': 20},
            2: {'zoom_range': 0.2},
            3: {}
        }
        selected_augumentation = augmentation_options.get(augmentation, {})

        normlization_options = {
            0: {'featurewise_center': True, 'featurewise_std_normalization': True},
            1: {'rescale': 1. / 255},
            2: {}
        }
        selected_normalization = normlization_options.get(normlization, {})

        test_image = self.load_image(self.val_images[self.current_idx])
        datagen = ImageDataGenerator(**selected_augumentation, **selected_normalization)

        pred_img = datagen.random_transform(test_image[0])
        pred_img = np.expand_dims(pred_img, axis=0)

        print("[DEBUG] Making prediction...")
        prediction = self.model.predict(pred_img)
        predicted_label = np.argmax(prediction)
        true_label = self.val_labels[self.current_idx]

        reward = self.prize if predicted_label == true_label else self.penality
        print(f"[DEBUG] Prediction: {predicted_label}, True Label: {true_label}, Reward: {reward}")

        self.current_idx += 1
        done = self.current_idx >= self.max_steps  # Stop after all images

        if not done:
            next_image = self.load_image(self.val_images[self.current_idx])
        else:
            next_image = np.zeros((128, 128, 1), dtype=np.uint8)  # Dummy final observation

        print(f"[DEBUG] Step completed. Next Index: {self.current_idx}, Done: {done}")
        return next_image, reward, done, False, {}

    def reset(self, seed=None, options=None):
        print("[DEBUG] Resetting environment...")
        self.current_idx = 0
        self.current_image = self.load_image(self.val_images[self.current_idx])
        return np.squeeze(self.current_image, axis=0), {}

    def render(self, mode='human'):
        pass


# Dummy data for validation images
val_images = [
    r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\0\ID00007637202177411956430_1.dcm.jpg',
    r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\1\ID00012637202177665765362_1.dcm.jpg',
    r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\1\ID00426637202313170790466_403.dcm.jpg',
    r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\2\ID00082637202201836229724_3.dcm.jpg',
    r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\2\ID00355637202295106567614_7.dcm.jpg'
]
val_labels = [0, 1, 1, 2, 2]

env = DLEnv(val_images, val_labels)
vec_env = make_vec_env(lambda: env, n_envs=1)

print("[DEBUG] Initializing PPO model...")
model = PPO("CnnPolicy", vec_env, verbose=1)
print("[DEBUG] Starting Training...")
model.learn(total_timesteps=100)
model.save("RL_agent")

print("[DEBUG] Loading trained agent...")
model = PPO.load("RL_agent")
env = DLEnv(val_images, val_labels)

obs, _ = env.reset()
print("[DEBUG] Environment reset done!")

# Running the trained model
for i in range(len(val_images)):
    print(f"[DEBUG] Step {i}")
    action, _states = model.predict(obs)
    print(f"[DEBUG] Action predicted: {action}")

    obs, reward, done, _, _ = env.step(action)
    print(f"[DEBUG] Step {i}: Reward {reward}, Done: {done}")

    if done:
        print("[DEBUG] Environment done, resetting...")
        #obs, _ = env.reset()
        break
