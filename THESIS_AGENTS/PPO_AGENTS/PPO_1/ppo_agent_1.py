import random
import os
import yaml
import numpy as np

from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import ProgressBarCallback


# Creating a custom callback on the model to save its logs every 1000 steps in tensorboard
class TensorBoardCallBack(BaseCallback):
    """
    Callback for saving a model, it is saved every ``check_freq`` steps

    :param check_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """

    def __init__(self, log_dir: str, check_freq: int, verbose=1):
        super(TensorBoardCallBack, self).__init__(verbose)
        self.log_dir = log_dir
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print("Saving latest model to {}".format(self.log_dir))
                # Save the agent
                self.model.save(os.path.join(
                    self.log_dir, f"check_point_{self.n_calls}"))
        return True


cfg_file = r"C:\Users\Usuario\Desktop\UNI\TESIS\THESIS_AGENTS\PPO_AGENTS\PPO_1\ppo_cfg_1.yaml"
with open(cfg_file, "r") as yaml_file:
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)

settings = params["settings"]
wrappers_settings = params["wrappers_settings"]
time_dep_seed = 42  # You can use any fixed or time-dependent seed value

# Creating the environment
env, num_envs = make_sb3_env(
    params["settings"]["game_id"], settings, wrappers_settings, seed=time_dep_seed)

agent = PPO("MultiInputPolicy", env, verbose=1,
            tensorboard_log=r"./ppo_1_tensorboard_logs/")

log_dir = r"./ppo_1_tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)
callback = TensorBoardCallBack(log_dir, 1000)
print("Policy architecture:")
print(agent.policy)
agent.learn(total_timesteps=1000, callback=callback, progress_bar=True)


# replace later the one with iteration of training
agent.save(f"./TRAINED_MODELS/ppo_1_model_{1}")


# Checking statement to see if theres a saved model
# If there is, load it
if os.path.exists("./TRAINED_MODELS/ppo_1_model_{1}.zip"):
    agent = PPO.load("./TRAINED_MODELS/ppo_1_model_{1}", env=env)


obs = env.reset()
cumulative_reward = 0.0

while True:
    env.render()
    action, _states = agent.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    cumulative_reward += reward

    if done:
        obs = env.reset()
        print("Cumulative reward:", cumulative_reward)
        cumulative_reward = 0.0
        break

env.close()
