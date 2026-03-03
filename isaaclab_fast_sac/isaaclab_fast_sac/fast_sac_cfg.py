from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class FastSacAlgorithmCfg:
    """Configuration for the FastSAC algorithm."""

    critic_learning_rate: float = 3e-4
    """The learning rate of the critic."""

    actor_learning_rate: float = 3e-4
    """The learning rate for the actor."""

    alpha_learning_rate: float = 3e-4
    """The learning rate for the alpha."""

    buffer_size: int = 1024
    """The replay memory buffer size per environment."""

    num_steps: int = 1
    """The number of steps to use for multi-step returns."""

    gamma: float = 0.97
    """The discount factor gamma."""

    tau: float = 0.125
    """Target smoothing coefficient."""

    batch_size: int = 8192
    """The batch size of sample from the replay memory."""

    learning_starts: int = 10
    """Timestep to start learning."""

    policy_frequency: int = 4
    """The frequency of training policy (delayed)."""

    num_updates: int = 8
    """The number of updates to perform per step."""

    target_entropy_ratio: float = 0.0
    """The ratio of the target entropy to the number of actions."""

    num_atoms: int = 101
    """The number of atoms for distributional critic."""

    v_min: float = -20.0
    """The minimum value of the support."""

    v_max: float = 20.0
    """The maximum value of the support."""

    critic_hidden_dim: int = 768
    """The hidden dimension of the critic network."""

    actor_hidden_dim: int = 512
    """The hidden dimension of the actor network."""

    alpha_init: float = 0.001
    """The initial value of the alpha."""

    use_autotune: bool = True
    """Whether to use autotune for the alpha."""

    use_tanh: bool = True
    """Whether to use tanh for the action."""

    log_std_max: float = 0.0
    """The maximum value of the log std."""

    log_std_min: float = -5.0
    """The minimum value of the log std."""

    compile: bool = True
    """Whether to use torch.compile."""

    obs_normalization: bool = True
    """Whether to enable observation normalization."""

    use_layer_norm: bool = True
    """Whether to use layer normalization."""

    num_q_networks: int = 2
    """Number of Q-networks to ensemble."""

    max_grad_norm: float = 0.0
    """The maximum gradient norm. 0 means no clipping."""

    amp: bool = True
    """Whether to use automatic mixed precision."""

    amp_dtype: str = "bf16"
    """The dtype for AMP (bf16 or fp16)."""

    weight_decay: float = 0.001
    """The weight decay of the optimizer."""

    logging_interval: int = 100
    """The interval to log the metrics."""

    action_scale: float | list[float] | None = None
    """Action scaling factor. If None, defaults to 1.0."""

    action_bias: float | list[float] | None = None
    """Action bias. If None, defaults to 0.0."""


@configclass
class FastSacRunnerCfg:
    """Configuration of the runner for FastSAC."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name.
    """

    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation sets to observation groups provided by the environment.

    Example::

        obs_groups = {
            "policy": ["policy"],
            "critic": ["policy", "privileged"],
        }
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done."""

    resume: bool = False
    """Whether to resume a previous training. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    algorithm: FastSacAlgorithmCfg = FastSacAlgorithmCfg()
    """The algorithm configuration."""
