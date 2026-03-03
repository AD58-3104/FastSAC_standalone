# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint from FastSAC agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a FastSAC checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--log_data", action="store_true", default=False, help="Enable data logging.")
parser.add_argument("--log_steps", type=int, default=5000, help="Maximum number of steps to log.")
parser.add_argument("--log_env_ids", type=str, default="0", help="Comma-separated environment IDs to log (e.g., '0,1,2').")
parser.add_argument("--log_output_dir", type=str, default="review/play_data", help="Directory to save logged data.")
# append FastSAC cli arguments
cli_args.add_fast_sac_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import time
import torch
from tqdm import tqdm

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# Import FastSAC components
from isaaclab_fast_sac import FastSacRunner, FastSacRunnerCfg, FastSacVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# import your.tasks  # noqa: F401


# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with FastSAC agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: FastSacRunnerCfg = cli_args.parse_fast_sac_cfg(args_cli.task, args_cli)

    # Set agent device if specified
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "fast_sac", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("fast_sac", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for FastSAC
    env = FastSacVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # initialize data logger if enabled
    logger = None
    if args_cli.log_data:
        # Parse environment IDs
        env_ids = [int(x.strip()) for x in args_cli.log_env_ids.split(",")]
        # Get the unwrapped environment for logging
        unwrapped_env = env.unwrapped
        # Unwrap RecordVideo wrapper if present
        if args_cli.video:
            unwrapped_env = unwrapped_env.env
        print(f"[INFO] Data logging enabled for environments: {env_ids}")
        print(f"[INFO] Max steps to log: {args_cli.log_steps}")
        print(f"[INFO] Output directory: {args_cli.log_output_dir}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    fast_sac_runner = FastSacRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    fast_sac_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = fast_sac_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # save logged data
    if logger is not None and logger.step_count > 0:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.abspath(args_cli.log_output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save as pickle
        pkl_path = os.path.join(output_dir, f"robot_data_{timestamp}.pkl")
        logger.save(pkl_path)

        # Save as CSV
        csv_dir = os.path.join(output_dir, f"csv_{timestamp}")
        logger.save_csv(csv_dir)

        print(f"[INFO] Data logging complete. Logged {logger.step_count} steps.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
