from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig()  
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  
config = config.resources(num_gpus=0)  
config = config.rollouts(num_rollout_workers=4)  
print(config.to_dict())  
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env="CartPole-v1")  
algo.train()  

"""
        algo = (
        PPOConfig()
        .resources(
            num_gpus=1
            )
        .rollouts(
            num_rollout_workers=1,
            enable_connectors=True,
            batch_mode="truncate_episodes",
            )
        .framework(
            framework="torch",
            )
        .environment(
            env="gt-rtgym-env-v1",
            disable_env_checking=True,
            render_env=False,
            )
        
        .training(
            train_batch_size=155,
        )
    
    """