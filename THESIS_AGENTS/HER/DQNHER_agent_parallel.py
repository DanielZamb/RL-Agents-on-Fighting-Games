import random
import os
import yaml
import numpy as np

from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import HerReplayBuffer, DQN
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
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



def train_evaluate_agent():

    # Loading the parameters from the yaml file
    cfg_file = r"C:\Users\T_STREET_FIGHTER\Desktop\TESIS\RL-Agents-on-Fighting-Games\THESIS_AGENTS\HER\DQNHER_cfg_parallel.yaml"
    with open(cfg_file, "r") as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)

    settings = params["settings"]
    wrappers_settings = params["wrappers_settings"]
    time_dep_seed = 42  # You can use any fixed or time-dependent seed value
    model_class = DQN


    # Creating the environment
    env, num_envs = make_sb3_env(
        params["settings"]["game_id"],settings, wrappers_settings, seed=time_dep_seed)

    log_dir = r"C:\Users\T_STREET_FIGHTER\Desktop\TESIS\RL-Agents-on-Fighting-Games\THESIS_AGENTS\HER\saved logs"

    os.makedirs(log_dir, exist_ok=True)

    print("Activated {} environment(s)".format(num_envs))

    print("Observation space =", env.observation_space)
    print("Act_space =", env.action_space)


    
    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    learning_rate = linear_schedule(
        ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    # clip_range = linear_schedule(
    #     ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    clip_range_vf = ppo_settings["clip_range"][1]
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]

    goal_selection_strategy="future"
    flagInit = False # -- cambiar despues de iniciar
    lastone = r"C:\Users\T_STREET_FIGHTER\Desktop\TESIS\RL-Agents-on-Fighting-Games\THESIS_AGENTS\HER\saved logs\check_point_0M\autosave_\2499996.zip"
    if os.path.exists(lastone) and flagInit:
        agent = DQN.load(lastone, 
                         env, verbose=1,
                    #replay_buffer_class=HerReplayBuffer,
                    gamma=gamma, 
                    learning_rate=learning_rate,
                    buffer_size=250000,
                    # replay_buffer_kwargs=dict(
                    # max_episode_length=2048, 
                    # n_sampled_goal=4,
                    # goal_selection_strategy=goal_selection_strategy
                    # ),
                    #optimize_memory_usage=True,
                    max_grad_norm=clip_range_vf,
                    policy_kwargs=policy_kwargs, 
                    tensorboard_log=log_dir)
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Loaded model from {lastone}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    else:
        agent = DQN("MultiInputPolicy", 
                    env, verbose=1,
                    #replay_buffer_class=HerReplayBuffer,
                    gamma=gamma, 
                    learning_rate=learning_rate,
                    buffer_size=250000,
                    # replay_buffer_kwargs=dict(
                    # max_episode_length=2048, 
                    # n_sampled_goal=4,
                    # goal_selection_strategy=goal_selection_strategy
                    # ),
                    #optimize_memory_usage=True,
                    max_grad_norm=clip_range_vf,
                    policy_kwargs=policy_kwargs, 
                    tensorboard_log=log_dir)

    print("Policy architecture:")
    print(agent.policy)

    check_freq = ppo_settings["autosave_freq"]
    auto_save_callback = AutoSave(check_freq=2500000, num_envs=num_envs,
                                  save_path=os.path.join(log_dir, f"check_point_0M_atbtn"))

    # Train the agennt
    time_steps = ppo_settings["time_steps"]
    agent.learn(total_timesteps=time_steps, callback=auto_save_callback)

    # replace later the one with iteration of training
    agent.save(f"C:/Users/T_STREET_FIGHTER/Desktop/TESIS/RL-Agents-on-Fighting-Games/THESIS_AGENTS\HER/TRAINED_MODELS/DQNHER_parallel_model_{50}M")

    # Checking statement to see if theres a saved model
    # If there is, load it
    if os.path.exists(f"C:/Users/T_STREET_FIGHTER/Desktop/TESIS/RL-Agents-on-Fighting-Games/THESIS_AGENTS/HER/TRAINED_MODELS/DQNHER_parallel_model_{50}M"):
        agent = DQN.load(
            f"C:/Users/T_STREET_FIGHTER/Desktop/TESIS/RL-Agents-on-Fighting-Games/THESIS_AGENTS/HER/TRAINED_MODELS/DQNHER_parallel_model_{50}M", env=env)

    # observation = env.reset()
    # cumulative_reward = [0.0 for _ in range(num_envs)]
    # while True:
    #     env.render()

    #     action, _state = agent.predict(observation, deterministic=True)

    #     observation, reward, done, info = env.step(action)
    #     cumulative_reward += reward
    #     if any(x != 0 for x in reward):
    #         print("Cumulative reward(s) =", cumulative_reward)

    #     if done.any():
    #         observation = env.reset()
    #         break

    # env.close()


if __name__ == "__main__":
    train_evaluate_agent()
