"""Standalone FastSAC implementation for IsaacLab."""

from .networks import Actor, Critic, DistributionalQNetwork
from .normalization import EmpiricalNormalization
from .replay_buffer import SimpleReplayBuffer

# These require isaaclab / rsl_rl to be installed and are imported lazily
# to allow using the core modules (networks, normalization, replay_buffer) standalone.


def __getattr__(name):
    if name == "FastSacAlgorithmCfg":
        from .fast_sac_cfg import FastSacAlgorithmCfg

        return FastSacAlgorithmCfg
    if name == "FastSacRunnerCfg":
        from .fast_sac_cfg import FastSacRunnerCfg

        return FastSacRunnerCfg
    if name == "FastSacRunner":
        from .fast_sac_runner import FastSacRunner

        return FastSacRunner
    if name == "FastSacVecEnvWrapper":
        from .vecenv_wrapper import FastSacVecEnvWrapper

        return FastSacVecEnvWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FastSacAlgorithmCfg",
    "FastSacRunnerCfg",
    "FastSacRunner",
    "FastSacVecEnvWrapper",
    "Actor",
    "Critic",
    "DistributionalQNetwork",
    "EmpiricalNormalization",
    "SimpleReplayBuffer",
]
