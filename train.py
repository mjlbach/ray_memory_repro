import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray_repro.model import ComplexInputNetwork
from ray_repro.debugging_env import RandomLargeObsSpaceEnv
#
# import ig_navigation
# from ig_navigation.callbacks import DummyCallback
# from ig_navigation.model import ComplexInputNetwork
# from ig_navigation.debugging_env import RandomLargeObsSpaceEnv

# Add a custom example feature extractor

ModelCatalog.register_custom_model("complex_input_network", ComplexInputNetwork)
model_cfg = {
    "post_fcnet_hiddens": [ 128, 128, 128],
    "custom_model": "complex_input_network",
    "conv_filters": [[16, [4, 4], 4], [32, [4, 4], 4], [256, [8, 8], 2]],
}

def main():
    ray.init()

    config = {
        "env": RandomLargeObsSpaceEnv,
        "model": model_cfg,
        "num_workers": 8,
        "framework": "torch",
        "seed": 0,
        "lambda": 0.9,
        "lr": 1e-4,
        "train_batch_size": 4096,
        "rollout_fragment_length": 4096/8,
        "num_sgd_iter": 30,
        "sgd_minibatch_size": 128,
        "gamma": 0.99,
        "create_env_on_driver": False,
        "num_gpus": 1,
    }

    trainer = ppo.PPOTrainer(config)

    for _ in range(1000):
        trainer.train()

if __name__ == "__main__":
    main()
