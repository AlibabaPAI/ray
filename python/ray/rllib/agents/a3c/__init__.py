from ray.rllib.agents.a3c.a3c import A3CTrainer, DEFAULT_CONFIG
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.utils import renamed_class

A2CAgent = renamed_class(A2CTrainer)
A3CAgent = renamed_class(A3CTrainer)

__all__ = [
    "A2CAgent", "A3CAgent", "A2CTrainer", "A3CTrainer", "DEFAULT_CONFIG"
]
