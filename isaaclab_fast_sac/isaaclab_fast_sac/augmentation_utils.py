from __future__ import annotations

from typing import Any

import torch


class SymmetryUtils:
    """Symmetry augmentation utilities for IsaacLab manager-based locomotion tasks."""

    def __init__(self, env: Any, obs_groups: dict[str, list[str]]) -> None:
        self.wrapper_env = env
        self.env = getattr(env, "unwrapped", env)
        self.obs_groups = obs_groups

        self.history_lengths: dict[str, int] = {}
        self.observation_dims: dict[str, int] = {}
        self.sub_observation_keys: dict[str, list[str]] = {}
        self.term_slices: dict[str, dict[str, tuple[int, int]]] = {}
        self.term_history_lengths: dict[str, dict[str, int]] = {}
        self.term_single_frame_dims: dict[str, dict[str, int]] = {}

        self.dof_index_map = torch.empty(0, dtype=torch.long)
        self.dof_sign_flip_mask = torch.empty(0)
        self.action_index_map = torch.empty(0, dtype=torch.long)
        self.action_sign_flip_mask = torch.empty(0)

        self._init_observation_config()
        self._init_joint_mappings()

    def _init_observation_config(self) -> None:
        obs_manager = getattr(self.env, "observation_manager", None)
        if obs_manager is None:
            raise ValueError("Symmetry augmentation requires an environment with an observation_manager.")

        active_terms = obs_manager.active_terms
        term_dims = obs_manager.group_obs_term_dim
        term_cfgs = obs_manager._group_obs_term_cfgs

        for group_name, group_term_names in active_terms.items():
            self.sub_observation_keys[group_name] = list(group_term_names)
            self.term_slices[group_name] = {}
            self.term_history_lengths[group_name] = {}
            self.term_single_frame_dims[group_name] = {}

            offset = 0
            for term_name, dims, term_cfg in zip(group_term_names, term_dims[group_name], term_cfgs[group_name], strict=True):
                full_dim = int(dims[-1]) if len(dims) > 0 else 1
                history_length = int(getattr(term_cfg, "history_length", 0) or 0)
                flatten_history_dim = bool(getattr(term_cfg, "flatten_history_dim", True))
                effective_history_length = max(history_length, 1)

                if history_length > 0 and not flatten_history_dim:
                    raise ValueError(
                        f"Symmetry augmentation requires flattened history for term '{group_name}/{term_name}'."
                    )

                if full_dim % effective_history_length != 0:
                    raise ValueError(
                        f"Observation term '{group_name}/{term_name}' has incompatible history shape: {dims}."
                    )

                self.term_slices[group_name][term_name] = (offset, offset + full_dim)
                self.term_history_lengths[group_name][term_name] = effective_history_length
                self.term_single_frame_dims[group_name][term_name] = full_dim // effective_history_length
                offset += full_dim

            self.observation_dims[group_name] = offset
            self.history_lengths[group_name] = 1

    def _init_joint_mappings(self) -> None:
        robot = self.env.scene["robot"]
        dof_names = list(robot.joint_names)
        action_term = self.env.action_manager._terms["joint_pos"]
        action_names = list(action_term._joint_names)

        self.dof_index_map, self.dof_sign_flip_mask = self._build_joint_mapping(dof_names)
        self.action_index_map, self.action_sign_flip_mask = self._build_joint_mapping(action_names)

    def _build_joint_mapping(self, joint_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

        index_map = []
        sign_mask = []
        for name in joint_names:
            mirrored_name = self._mirror_joint_name(name)
            index_map.append(name_to_idx.get(mirrored_name, name_to_idx[name]))
            sign_mask.append(-1.0 if self._should_flip_joint_sign(name) else 1.0)

        return (
            torch.tensor(index_map, device=self.env.device, dtype=torch.long),
            torch.tensor(sign_mask, device=self.env.device, dtype=torch.float),
        )

    @staticmethod
    def _mirror_joint_name(name: str) -> str:
        replacements = (
            ("ALeft_", "ARight_"),
            ("ARight_", "ALeft_"),
            ("Left_", "Right_"),
            ("Right_", "Left_"),
            ("left_", "right_"),
            ("right_", "left_"),
            ("left", "right"),
            ("right", "left"),
        )
        for src, dst in replacements:
            if src in name:
                return name.replace(src, dst, 1)
        return name

    @staticmethod
    def _should_flip_joint_sign(name: str) -> bool:
        lower_name = name.lower()
        return "yaw" in lower_name or "roll" in lower_name

    def augment_observations(self, obs: torch.Tensor, obs_list: list[str]) -> torch.Tensor:
        mirrored_obs = self.mirror_xz_plane(obs, obs_list)
        return torch.cat((obs, mirrored_obs), dim=0)

    def augment_actions(self, actions: torch.Tensor) -> torch.Tensor:
        mirrored_actions = actions[..., self.action_index_map] * self.action_sign_flip_mask
        return torch.cat((actions, mirrored_actions), dim=0)

    def mirror_xz_plane(self, observation: torch.Tensor, obs_list: list[str]) -> torch.Tensor:
        mirrored_obs_all = observation.clone()
        batch_size = mirrored_obs_all.shape[0]
        group_offset = 0

        for group_name in obs_list:
            cur_obs_length = self.observation_dims[group_name]
            mirrored_obs = mirrored_obs_all[..., group_offset : group_offset + cur_obs_length].clone()
            for sub_obs_key in self.sub_observation_keys[group_name]:
                mirror_fn = getattr(self, f"mirror_obs_{sub_obs_key}", None)
                if mirror_fn is None:
                    raise ValueError(f"Unsupported symmetry observation term: {sub_obs_key}")
                start, end = self.term_slices[group_name][sub_obs_key]
                history_length = self.term_history_lengths[group_name][sub_obs_key]
                single_frame_dim = self.term_single_frame_dims[group_name][sub_obs_key]
                term_obs = mirrored_obs[..., start:end].reshape(batch_size, history_length, single_frame_dim).clone()
                mirrored_obs[..., start:end] = mirror_fn(term_obs).reshape(batch_size, end - start)

            mirrored_obs_all[..., group_offset : group_offset + cur_obs_length] = mirrored_obs
            group_offset += cur_obs_length

        return mirrored_obs_all

    @staticmethod
    def mirror_obs_base_lin_vel(base_lin_vel: torch.Tensor) -> torch.Tensor:
        base_lin_vel[..., 1] = -base_lin_vel[..., 1]
        return base_lin_vel

    @staticmethod
    def mirror_obs_base_ang_vel(base_ang_vel: torch.Tensor) -> torch.Tensor:
        base_ang_vel[..., 0] = -base_ang_vel[..., 0]
        base_ang_vel[..., 2] = -base_ang_vel[..., 2]
        return base_ang_vel

    @staticmethod
    def mirror_obs_projected_gravity(projected_gravity: torch.Tensor) -> torch.Tensor:
        projected_gravity[..., 1] = -projected_gravity[..., 1]
        return projected_gravity

    @staticmethod
    def mirror_obs_velocity_commands(velocity_commands: torch.Tensor) -> torch.Tensor:
        if velocity_commands.shape[-1] >= 2:
            velocity_commands[..., 1] = -velocity_commands[..., 1]
        if velocity_commands.shape[-1] >= 3:
            velocity_commands[..., 2] = -velocity_commands[..., 2]
        if velocity_commands.shape[-1] >= 4:
            velocity_commands[..., 3] = -velocity_commands[..., 3]
        return velocity_commands

    def mirror_obs_joint_pos(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return joint_pos[..., self.dof_index_map] * self.dof_sign_flip_mask

    def mirror_obs_joint_vel(self, joint_vel: torch.Tensor) -> torch.Tensor:
        return joint_vel[..., self.dof_index_map] * self.dof_sign_flip_mask

    def mirror_obs_last_action(self, last_action: torch.Tensor) -> torch.Tensor:
        return last_action[..., self.action_index_map] * self.action_sign_flip_mask

    @staticmethod
    def mirror_obs_sincos_phase(sincos_phase: torch.Tensor) -> torch.Tensor:
        if sincos_phase.shape[-1] != 4:
            raise ValueError(f"Expected sincos_phase to have size 4, got {sincos_phase.shape[-1]}.")
        return sincos_phase[..., [1, 0, 3, 2]]
