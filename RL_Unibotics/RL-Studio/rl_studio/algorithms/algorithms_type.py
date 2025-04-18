from enum import Enum


class AlgorithmsType(Enum):
    DDPG_2 = "ddpg_2"
    PROGRAMMATIC = 'programmatic'
    QLEARN = "qlearn"
    DEPRECATED_QLEARN = "qlearn_deprecated"
    DQN = "dqn"
    DDPG = "ddpg"
    SAC = "sac"
    SAC_2 = "sac_2"
    DDPG_TORCH = "ddpg_torch"
    PPO = "ppo"
    PPO_CONTINIUOUS = 'ppo_continuous'
    MANUAL = "manual"
    AUTO = "auto"
