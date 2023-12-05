import filtros
filtros.activate()
from ray.rllib.algorithms.ppo import PPOConfig

print('========================================================================')
config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("Taxi-v3")
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()  # 2. build the algorithm,

for _ in range(2):
    algo.train()  # 3. train it,

print('evaluate')
algo.evaluate()  # 4. and evaluate it.