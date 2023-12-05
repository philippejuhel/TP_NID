
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import shutil
import os

CHECKPOINT_ROOT = "tmp/ppo/taxi"
SELECT_ENV = "Taxi-v3"
N_ITER = 30 # Nb of iterations to train 

# Configure checkpoint folder to store results
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
ray_results = os.getcwd() + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# Configure Proximal Policy Optimization Algorithm
config = PPOConfig()  
config.debugging(log_level="WARN")
algo = config.build(env=SELECT_ENV)

ray.shutdown()
context = ray.init(ignore_reinit_error=True)
print(context.dashboard_url)  # Print URL and port to access Ray Dashboard

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
  result = algo.train()
  file_name = algo.save(CHECKPOINT_ROOT)

  print(s.format(
    n + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"],
    file_name
   ))
  
  