import diambra.arena
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config
from ray.rllib.algorithms.a3c import A3C
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print

class AutoSave(DefaultCallbacks):
    def on_train_result(self, trainer, result):
        step = result["training_iteration"]
        if step % 50000 == 0:  # Save every 2 steps.
            checkpoint = trainer.save()
            print(f"Checkpoint saved at {checkpoint} at step {step}")



if __name__ == "__main__":

    # Settings
    settings = {}
    settings["hardcore"] = True
    settings["frame_shape"] = (84, 84, 1)
    settings["characters"] = "Ryu"
    settings["difficulty"] =  4
    settings["Settings"] = 4
    settings["action_space"] = "discrete"   
    settings["step_ratio"] = 6
    settings["continue_game"] = 0.0
    settings["attack_but_combination"] = True
    settings["show_final"] = True
    settings["player"] = "Random"
    settings["char_outfits"] = 2

    # Wrappers Settings
    wrappers_settings = {}
    wrappers_settings["reward_normalization"] = True
    wrappers_settings["actions_stack"] = 12
    wrappers_settings["frame_stack"] = 6
    wrappers_settings["scale"] = True
    wrappers_settings["dilation"] = 1
    wrappers_settings["scale"] = True
    wrappers_settings["exclude_image_scaling"] = True
    wrappers_settings["flatten"] = True
    wrappers_settings["filter_keys"] = ["stage", "P1_ownHealth","P1_oppHealth","P1_ownSide","P1_oppSide","P1_ownChar",
                                        "P1_oppChar","P1_actions_move","P1_actions_attack","P1_ownStunBar","P1_oppStunBar",
                                        "P1_ownWins","P1_oppWins","P1_ownStunned","P1_oppStunned","P1_ownSuperBar","P1_oppSuperBar",
                                        "P1_ownSuperType","P1_oppSuperType","P1_ownSuperCount","P1_oppSuperCount","P1_ownSuperMax",
                                        "P1_oppSuperMax",]


    config = {
        # Define and configure the environment
        "env": DiambraArena,
        "env_config": {
            "game_id": "sfiii3n",
            "settings": settings,
            "wrappers_settings": wrappers_settings,
        },
        "train_batch_size": 200,
        # Use 2 rollout workers
        "num_workers": 5,
        # Use a vectorized env with 2 sub-envs.
        "num_envs_per_worker": 5,
        # Evaluate once per training iteration.
        "evaluation_interval": 1,
        # Run evaluation on (at least) two episodes
        "evaluation_duration": 2,
        # ... using one evaluation worker (setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        "evaluation_num_workers": 1,
        "framework": "torch",
        "callbacks": AutoSave,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            "render_env": True,
        },
    }

    # Update config file
    config = preprocess_ray_config(config)

    # Create the RLlib Agent.
    agent = A3C(config=config)

    # Run it for n training iterations
    print("\nStarting training ...\n")
    for idx in range(30000000):
        print("Training iteration:", idx + 1)
        results = agent.train()
    print("\n .. training completed.")
    print("Training results:\n{}".format(pretty_print(results)))
