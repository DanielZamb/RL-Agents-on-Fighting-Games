import random
import os
import yaml
import numpy as np

from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule

from pathlib import Path


class AutoSave(BaseCallback):
    """
    Callback for saving a model, it is saved every ``check_freq`` steps

    :param check_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, num_envs: int, save_path: str, verbose=1):
        super(AutoSave, self).__init__(verbose)
        self.check_freq = int(check_freq / num_envs)
        self.num_envs = num_envs
        self.save_path_base = os.path.join(save_path, "autosave_")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print("Saving latest model to {}".format(self.save_path_base))
            # Save the agent
            self.model.save(
                os.path.join(self.save_path_base, f"{str(self.n_calls * self.num_envs)}"))

        return True


# Loading the parameters from the yaml file
cfg_file = r"C:\Users\USER\Desktop\UNI\TESIS\RL-Agents-on-Fighting-Games\THESIS_AGENTS\PPO_AGENTS\PPO_1\ppo_cfg_1.yaml"
with open(cfg_file, "r") as yaml_file:
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)

settings = params["settings"]
wrappers_settings = params["wrappers_settings"]
time_dep_seed = 42  # You can use any fixed or time-dependent seed value

# Creating the environment
env, num_envs = make_sb3_env(
    params["settings"]["game_id"], settings, wrappers_settings, seed=time_dep_seed)

log_dir = r".\ppo_1_tensorboard_logs"
agent = PPO("MultiInputPolicy", env, verbose=1,
            tensorboard_log=log_dir, n_steps=2048, batch_size=64, n_epochs=10)


os.makedirs(log_dir, exist_ok=True)
# callback = TensorBoardCallBack(log_dir, 1000)
print("Policy architecture:")
print(agent.policy)

auto_save_callback = AutoSave(check_freq=100000, num_envs=num_envs,
                              save_path=os.path.join(log_dir, f"check_point_"))

# Train the agennt
agent.learn(total_timesteps=2000000,
            callback=auto_save_callback, progress_bar=True)

# replace later the one with iteration of training
agent.save(f"./TRAINED_MODELS/ppo_1_model_{2}M")


# Checking statement to see if theres a saved model
# If there is, load it
if os.path.exists("./TRAINED_MODELS/ppo_1_model_{2}M.zip"):
    agent = PPO.load("./TRAINED_MODELS/ppo_1_model_{2}M", env=env)


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
