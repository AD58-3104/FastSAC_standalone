from __future__ import annotations

import math
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import tqdm
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import nn, optim
from torch.amp import GradScaler, autocast

from .networks import Actor, Critic
from .normalization import EmpiricalNormalization
from .replay_buffer import SimpleReplayBuffer

torch.set_float32_matmul_precision("high")


class FastSacRunner:
    """Training runner for FastSAC with IsaacLab environments.

    FastSAC is an efficient variant of Soft Actor-Critic (SAC) tuned for
    large-scale training with massively parallel simulation.
    See https://arxiv.org/abs/2505.22642 for more details about FastTD3.

    Follows the same runner pattern as rsl_rl's ``OnPolicyRunner``.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env = env

        # configure multi-GPU
        self._configure_multi_gpu()

        # store training configuration
        self.save_interval = self.cfg["save_interval"]

        # query observations from environment
        obs = self.env.get_observations()
        self.obs_groups = self.cfg["obs_groups"]

        # set up the algorithm
        self._setup(obs)

        # decide whether to disable logging (only log from rank 0)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # logging
        self.log_dir = log_dir
        self.writer = None
        self.current_learning_iteration = 0
        self.global_step = 0

    def _setup(self, obs: TensorDict) -> None:
        """Set up networks, optimizers, replay buffer and normalizers."""
        alg = self.alg_cfg
        device = self.device

        # Compute obs dimensions from obs_groups
        policy_groups = self.obs_groups.get("policy", [])
        critic_groups = self.obs_groups.get("critic", [])

        self.num_actor_obs = sum(obs[g].shape[-1] for g in policy_groups)
        self.num_critic_obs = sum(obs[g].shape[-1] for g in critic_groups)
        num_actions = self.env.num_actions

        # Build action scale/bias tensors
        action_scale = self._build_action_param(alg.get("action_scale"), num_actions, default=1.0)
        action_bias = self._build_action_param(alg.get("action_bias"), num_actions, default=0.0)

        # Create actor
        self.actor = Actor(
            num_obs=self.num_actor_obs,
            num_actions=num_actions,
            hidden_dim=alg["actor_hidden_dim"],
            log_std_max=alg["log_std_max"],
            log_std_min=alg["log_std_min"],
            use_tanh=alg["use_tanh"],
            use_layer_norm=alg["use_layer_norm"],
            device=device,
            action_scale=action_scale,
            action_bias=action_bias,
        )

        # Create critic (Q-networks)
        self.qnet = Critic(
            num_obs=self.num_critic_obs,
            num_actions=num_actions,
            num_atoms=alg["num_atoms"],
            v_min=alg["v_min"],
            v_max=alg["v_max"],
            hidden_dim=alg["critic_hidden_dim"],
            use_layer_norm=alg["use_layer_norm"],
            num_q_networks=alg["num_q_networks"],
            device=device,
        )

        # Create target critic
        self.qnet_target = Critic(
            num_obs=self.num_critic_obs,
            num_actions=num_actions,
            num_atoms=alg["num_atoms"],
            v_min=alg["v_min"],
            v_max=alg["v_max"],
            hidden_dim=alg["critic_hidden_dim"],
            use_layer_norm=alg["use_layer_norm"],
            num_q_networks=alg["num_q_networks"],
            device=device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Log alpha
        self.log_alpha = torch.tensor([math.log(alg["alpha_init"])], requires_grad=True, device=device)
        self.target_entropy = -num_actions * alg["target_entropy_ratio"]

        # Optimizers
        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=alg["critic_learning_rate"],
            weight_decay=alg["weight_decay"],
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=alg["actor_learning_rate"],
            weight_decay=alg["weight_decay"],
            fused=True,
            betas=(0.9, 0.95),
        )
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha], lr=alg["alpha_learning_rate"], fused=True, betas=(0.9, 0.95)
        )

        # GradScaler for AMP
        self.scaler = GradScaler(enabled=alg["amp"])

        # Observation normalizers
        self.obs_normalization = alg["obs_normalization"]
        if self.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=self.num_actor_obs, device=device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=self.num_critic_obs, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Replay buffer
        self.rb = SimpleReplayBuffer(
            n_env=self.env.num_envs,
            buffer_size=alg["buffer_size"],
            n_obs=self.num_actor_obs,
            n_act=num_actions,
            n_critic_obs=self.num_critic_obs,
            n_steps=alg["num_steps"],
            gamma=alg["gamma"],
            device=device,
        )

        # Synchronize model parameters across GPUs
        if self.is_distributed:
            self._synchronize_model_parameters()

    @staticmethod
    def _build_action_param(value, num_actions: int, default: float) -> torch.Tensor:
        """Convert action_scale/action_bias config value to a tensor."""
        if value is None:
            return torch.full((num_actions,), default)
        if isinstance(value, (int, float)):
            return torch.full((num_actions,), float(value))
        return torch.tensor(value, dtype=torch.float)

    def _concat_obs(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        """Concatenate observation groups into a flat tensor."""
        return torch.cat([obs[g] for g in groups], dim=-1)

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.alg_cfg["amp_dtype"] == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.alg_cfg["amp"]):
            yield

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop.

        Args:
            num_learning_iterations: Number of iterations to train.
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
        alg = self.alg_cfg
        device = self.device

        # Initialize logging
        self._prepare_logging_writer()

        # Randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Optionally compile
        if alg["compile"]:
            update_main = torch.compile(self._update_main)
            update_pol = torch.compile(self._update_pol)
            policy = torch.compile(self.actor.explore)
            normalize_obs = torch.compile(self.obs_normalizer.forward)
            normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        else:
            update_main = self._update_main
            update_pol = self._update_pol
            policy = self.actor.explore
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward

        qnet = self.qnet
        qnet_target = self.qnet_target
        rb = self.rb

        policy_groups = self.obs_groups["policy"]
        critic_groups = self.obs_groups["critic"]

        # Initial observation
        obs_td = self.env.get_observations().to(device)
        actor_obs = self._concat_obs(obs_td, policy_groups)
        critic_obs = self._concat_obs(obs_td, critic_groups)

        dones = None

        # Book keeping
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=device)
        reward_log_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        pbar = tqdm.tqdm(total=tot_iter, initial=start_iter, disable=self.disable_logs)

        for it in range(start_iter, tot_iter):
            self.global_step = it
            start = time.time()

            # --- Collection phase ---
            with torch.no_grad(), self._maybe_amp():
                norm_obs = normalize_obs(actor_obs, update=False)
                actions = policy(obs=norm_obs, dones=dones)

            obs_td, rewards, dones, infos = self.env.step(actions.float())
            next_actor_obs = self._concat_obs(obs_td, policy_groups)
            next_critic_obs = self._concat_obs(obs_td, critic_groups)

            # Track episode stats
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

            # Collect per-term reward logs from episodes that just ended
            if new_ids.numel() > 0 and "log" in infos:
                reset_ids = new_ids[:, 0]
                for key, value in infos["log"].items():
                    if isinstance(value, torch.Tensor):
                        if value.numel() == self.env.num_envs:
                            vals = value[reset_ids].cpu().float().tolist()
                        else:
                            vals = [value.mean().item()]
                    else:
                        vals = [float(value)]
                    reward_log_buffers[key].extend(vals)

            # IsaacLab auto-resets: obs_td is already post-reset for done envs.
            # For the replay buffer, we need pre-reset obs. We use time_outs from extras
            # to detect truncations vs true terminations.
            truncations = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))

            # Build transition for replay buffer
            # For terminal envs, next_obs in buffer should be the TRUE last obs (not post-reset).
            # However IsaacLab doesn't provide this directly in all cases.
            # We use truncation to mark bootstrap vs no-bootstrap.
            transition = TensorDict(
                {
                    "observations": actor_obs,
                    "actions": actions.float(),
                    "next": {
                        "observations": next_actor_obs,
                        "rewards": rewards.float(),
                        "truncations": truncations.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(self.env.num_envs,),
                device=device,
            )
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = next_critic_obs

            actor_obs = next_actor_obs
            critic_obs = next_critic_obs

            rb.extend(transition)

            collection_time = time.time() - start

            # --- Learning phase ---
            learn_time = 0.0
            batch_size = max(alg["batch_size"] // self.env.num_envs // self.gpu_world_size, 1)
            if it >= alg["learning_starts"]:
                learn_start = time.time()

                prepared_batches = self._sample_and_prepare_batches(
                    batch_size, alg["num_updates"], normalize_obs, normalize_critic_obs
                )
                for i, data in enumerate(prepared_batches):
                    (
                        buffer_rewards,
                        critic_grad_norm,
                        qf_loss,
                        qf_max,
                        qf_min,
                        alpha_loss,
                    ) = update_main(data)

                    if alg["num_updates"] > 1:
                        if i % alg["policy_frequency"] == 1:
                            actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)
                    elif it % alg["policy_frequency"] == 0:
                        actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)

                    # Update target network
                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - alg["tau"])
                        torch._foreach_add_(tgt_ps, src_ps, alpha=alg["tau"])

                learn_time = time.time() - learn_start

            # --- Logging ---
            if not self.disable_logs and it % alg["logging_interval"] == 0 and it > 0:
                self._log(it, rewbuffer, lenbuffer, collection_time, learn_time, locals(), reward_log_buffers)

            # --- Saving ---
            if (
                self.save_interval > 0
                and it > 0
                and it % self.save_interval == 0
                and not self.disable_logs
            ):
                self.save(os.path.join(self.log_dir, f"model_{it:07d}.pt"))

            self.current_learning_iteration = it + 1
            pbar.update(1)

        pbar.close()

        # Final save
        if not self.disable_logs and self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{tot_iter:07d}.pt"))

    def _update_main(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alg = self.alg_cfg

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                discount = alg["gamma"] ** data["next"]["effective_n_steps"]

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_distributed:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if alg["max_grad_norm"] > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=alg["max_grad_norm"],
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if alg["use_autotune"]:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_distributed:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

            scaler.unscale_(alpha_optimizer)
            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
        )

    def _update_pol(self, data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        alg = self.alg_cfg

        with self._maybe_amp():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_distributed:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)

        if alg["max_grad_norm"] > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=alg["max_grad_norm"],
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _sample_and_prepare_batches(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """Sample a large batch once and split into smaller batches for each update."""
        large_batch_size = batch_size * num_updates
        large_data = self.rb.sample(large_batch_size)
        samples_per_update = batch_size * self.env.num_envs

        # Normalize all data once
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        large_data["critic_observations"] = normalize_critic_obs(large_data["critic_observations"])
        large_data["next"]["critic_observations"] = normalize_critic_obs(large_data["next"]["critic_observations"])

        # Split into smaller batches
        prepared_batches = []

        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                    "critic_observations": large_data["critic_observations"][start_idx:end_idx],
                },
                batch_size=samples_per_update,
            )
            batch_data["next"]["critic_observations"] = large_data["next"]["critic_observations"][start_idx:end_idx]

            prepared_batches.append(batch_data)

        return prepared_batches

    def save(self, path: str) -> None:
        """Save model parameters and training state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        saved_dict = {
            "model_state_dict": {
                "actor": self.actor.state_dict(),
                "qnet": self.qnet.state_dict(),
                "qnet_target": self.qnet_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "obs_normalizer": self.obs_normalizer.state_dict() if self.obs_normalization else None,
                "critic_obs_normalizer": (
                    self.critic_obs_normalizer.state_dict() if self.obs_normalization else None
                ),
            },
            "optimizer_state_dict": {
                "actor": self.actor_optimizer.state_dict(),
                "q": self.q_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            "iter": self.current_learning_iteration,
            "config": self.cfg,
        }
        torch.save(saved_dict, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """Load model parameters and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model_sd = checkpoint["model_state_dict"]
        self.actor.load_state_dict(model_sd["actor"])
        self.qnet.load_state_dict(model_sd["qnet"])
        self.qnet_target.load_state_dict(model_sd["qnet_target"])
        self.log_alpha.data.copy_(model_sd["log_alpha"].to(self.device))

        if self.obs_normalization and model_sd.get("obs_normalizer") is not None:
            self.obs_normalizer.load_state_dict(model_sd["obs_normalizer"])
            self.critic_obs_normalizer.load_state_dict(model_sd["critic_obs_normalizer"])

        opt_sd = checkpoint["optimizer_state_dict"]
        self.actor_optimizer.load_state_dict(opt_sd["actor"])
        self.q_optimizer.load_state_dict(opt_sd["q"])
        self.alpha_optimizer.load_state_dict(opt_sd["alpha"])
        if opt_sd.get("scaler") is not None:
            self.scaler.load_state_dict(opt_sd["scaler"])

        self.current_learning_iteration = checkpoint.get("iter", 0)
        print(f"Loaded checkpoint from {path} (iteration {self.current_learning_iteration})")

    def get_inference_policy(self, device: str | None = None):
        """Return an inference policy function compatible with rsl_rl's runner interface.

        Args:
            device: Device to put the policy on.

        Returns:
            A callable that takes a TensorDict of observations and returns actions.
        """
        device = device or self.device
        actor = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        actor.eval()
        obs_normalizer.eval()
        obs_normalization = self.obs_normalization
        policy_groups = self.obs_groups["policy"]

        def policy_fn(obs: TensorDict) -> torch.Tensor:
            flat_obs = torch.cat([obs[g] for g in policy_groups], dim=-1)
            if obs_normalization:
                flat_obs = obs_normalizer(flat_obs, update=False)
            return actor(flat_obs)[0]

        return policy_fn

    def train_mode(self):
        """Set networks to train mode."""
        self.actor.train()
        self.qnet.train()
        if self.obs_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        """Set networks to eval mode."""
        self.actor.eval()
        self.qnet.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' >= world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' >= world size '{self.gpu_world_size}'."
            )

        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        torch.cuda.set_device(self.gpu_local_rank)

    def _synchronize_model_parameters(self):
        """Synchronize actor, qnet, and log_alpha parameters across all GPUs."""
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)
        torch.distributed.broadcast(self.log_alpha.data, src=0)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Batches and all-reduces gradients across GPUs."""
        if not self.is_distributed:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    def _prepare_logging_writer(self):
        """Initialize the TensorBoard writer."""
        if self.disable_logs or self.log_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        except ImportError:
            print("TensorBoard not available. Install tensorboard for logging support.")

    def _log(
        self,
        it: int,
        rewbuffer,
        lenbuffer,
        collection_time: float,
        learn_time: float,
        locs: dict,
        reward_log_buffers: dict | None = None,
    ):
        """Log training metrics."""
        if self.writer is None:
            return

        if len(rewbuffer) > 0:
            mean_reward = sum(rewbuffer) / len(rewbuffer)
            mean_length = sum(lenbuffer) / len(lenbuffer)
            self.writer.add_scalar("Episode/reward", mean_reward, it)
            self.writer.add_scalar("Episode/length", mean_length, it)

        if "qf_loss" in locs:
            self.writer.add_scalar("Loss/qf_loss", locs.get("qf_loss", torch.tensor(0.0)).item(), it)
        if "actor_loss" in locs:
            self.writer.add_scalar("Loss/actor_loss", locs.get("actor_loss", torch.tensor(0.0)).item(), it)
        if "alpha_loss" in locs:
            self.writer.add_scalar("Loss/alpha_loss", locs.get("alpha_loss", torch.tensor(0.0)).item(), it)

        self.writer.add_scalar("Alpha/value", self.log_alpha.exp().item(), it)
        self.writer.add_scalar("Perf/collection_time", collection_time, it)
        self.writer.add_scalar("Perf/learn_time", learn_time, it)

        # Log per-term reward statistics
        if reward_log_buffers:
            for key, buf in reward_log_buffers.items():
                if len(buf) > 0:
                    self.writer.add_scalar(f"Rewards/{key}", sum(buf) / len(buf), it)

        if len(rewbuffer) > 0:
            print(f"[{it}] reward: {mean_reward:.2f} | ep_len: {mean_length:.0f} | alpha: {self.log_alpha.exp().item():.4f}")
