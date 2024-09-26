from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env 
from Model_Enviorment import ModelEnv
env=ModelEnv()
check_env(env,warn=True)
model = DQN(
    'MlpPolicy', 
    env, 
    learning_rate=0.0005, 
    buffer_size=100000,               # Larger buffer size
    exploration_initial_eps=1.0,      # Start with more exploration
    exploration_fraction=0.3,         # Explore for 30% of timesteps
    exploration_final_eps=0.02,       # Minimum exploration rate
    gamma=0.95,                       # Discount factor for future rewards
    target_update_interval=10000,     # Update target network less frequently          # Use prioritized experience replay
    verbose=1
)
model.learn(total_timesteps=300000)
model.save('RL_fluAgent')
obs,info=env.reset()
for i in range(90):
    action,states=model.predict(obs)
    obs,reward,terminated,truncated,info=env.step(action)
    env.render(reward)
    if terminated:
        obs,_=env.reset()
    

 