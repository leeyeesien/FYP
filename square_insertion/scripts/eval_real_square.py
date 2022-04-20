"""Script for running evaluation on a real robot
"""
from learnmp.utils.samplers import BoundarySampler, DistanceSampler
import ray
from typing import Iterable
import json
import rich
import os
import numpy as np
import gym
import argparse
import multiprocessing as mp
from franka_cartesian_controller.monitor import MotionGeneratorMonitor
from learnmp.utils.garage_utils import load
from learnmp.utils.wrapper import (
    SequenceWrapper,
    ActionInputWrapper,
    StateWrapperRos,
)
import learnmp.utils.transform_utils2 as T
from learnmp.controllers.motion_primitives_list import (
    motion_primitive_list_real,
    motion_primitive_compact_real,
)
import learnmp
from garage.envs import GymEnv
from gym.utils import EzPickle

from scripts.insertion_seq import HOLE_FRAME_POS

import franka_cartesian_controller.motion_generator as motion_generator
import math


class GymGarageInterface(GymEnv, EzPickle):
    """
    `GymEnv` class in `garage` has some problem when saving `mujoco_py.MjSim`
    object. Therefore, only save env_id and config
    """

    def __init__(self, env_id, env_kwargs=None, *argv, **kwargs):
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, d):
        self.config = d

    def __del__(self):
        pass


learnmp.utils.env_utils.GymGarageInterface = GymGarageInterface


class Logger:
    def __init__(self, log_file_name=None) -> None:
        root_dir = os.path.dirname(os.path.dirname(learnmp.__file__))
        self.log_dir = os.path.join(root_dir, "exp/eval")
        self.save_path = os.path.join(self.log_dir, log_file_name)

    def print(self, msg):
        rich.print(msg)
        with open(self.save_path, "a") as f:
            f.write(msg)
            f.write("\n")


def evaluate_experiment(args_exp_path, args_config, Tgoal, args_log, args_init_pos_noise, args_init_ori_noise, args_est_pos_error, args_est_ori_error, args_eval_eps):

    robot = motion_generator.MotionGenerator(verbose="debug")

    # load experiment
    data = load(args_exp_path)
    ray.shutdown()
    print("test 2")
    # policy
    policy = data["algo"].policy
    # print(policy.batch_norm.state_dict())
    # initialize env
    with open(args_config, "r") as f:
        real_env_config = json.load(f)
    hole_depth = real_env_config["hole_depth"]
    # Tgoal = (
    #     np.array(real_env_config["goal_pose_transformation"]).reshape((4, 4)).T
    # )
    Tgoal = T.make_pose(Tgoal[:3], T.quat2mat(Tgoal[3:]))
    rel = np.array([np.pi, 0, 0])

    goal_quat = T.mat2quat(Tgoal[:3, :3])
    hole_frame_quat = T.quat_multiply(goal_quat, T.axisangle2quat(rel))
    # hole_frame_pos = Tgoal[:3, 3] + T.quat2mat(hole_frame_quat).dot(np.array([0, 0, hole_depth]))
    hole_frame_pos = Tgoal[:3, 3]

    #
    init_args = data["env"].config["_ezpickle_args"]
    env_id = init_args[0]
    env_config = init_args[-1]
    seq_length = init_args[1]
    state_wrapper = False
    action_input = False
    if len(init_args) == 5:
        if init_args[3]:
            state_wrapper = True
        if init_args[2]:
            action_input = True
    elif len(init_args) == 4:
        if init_args[2]:
            action_input = True
    print(seq_length)

    # initialization_noise = env_config["initialization_noise"]
    # pose_estimation_error = env_config["pose_estimation_error"]
    initialization_noise = (0, 0.0)
    pose_estimation_error = (0, 0.0)
    horizon = env_config["horizon"]
    state_eef_base = env_config.get("state_eef_base", False)
    # check for hybrid env
    if "Hybrid" not in env_id:
        # env_id = "learnmp:PegInHolePandaRos-v1"
        env_id = "PegInHolePandaRos-v1"
        # motion primitive
        mp_list, _ = motion_primitive_list_real()
    else:
        # env_id = "learnmp:PegInHoleHybridPandaRos-v1"
        env_id = "PegInHoleHybridPandaRos-v1"
        mp_list, _ = motion_primitive_compact_real()

    # evaluate on the boundary condition
    initialization_noise_sampler = DistanceSampler(
        initialization_noise[0] + args_init_pos_noise / 1000,
        initialization_noise[1] + args_init_ori_noise / 180 * np.pi,
    )
    pose_estimation_error_sampler = DistanceSampler(
        pose_estimation_error[0] + args_est_pos_error / 1000,
        pose_estimation_error[1] + args_est_ori_error / 180 * np.pi,
    )

    # TODO test evaluation for different env
    env = gym.make(
        env_id,
        hole_frame_pos=hole_frame_pos,  
        hole_depth=hole_depth,
        hole_frame_quat=hole_frame_quat, 
        motion_primitive_list=mp_list,
        initialization_noise=initialization_noise,
        pose_estimation_error=pose_estimation_error,
        initialization_noise_sampler=initialization_noise_sampler,
        pose_estimation_error_sampler=pose_estimation_error_sampler,
        horizon=horizon,
        state_eef_base=state_eef_base,
    )

    # wrapper
    # obs_dim = policy._obs_dim
    if seq_length > 1:
        env = SequenceWrapper(env, seq_length=seq_length)

    if action_input:
        env = ActionInputWrapper(env)
    if state_wrapper:
        env = StateWrapperRos(env)

    # log
    if args_log:
        from datetime import datetime

        now = datetime.now()
        eval_exp_name = args_exp_path.split("/")
        log_file_name = (
            eval_exp_name[-1] + "-eval" + now.strftime("-%d-%m-%H-%M")
        )
        logger = Logger(log_file_name + ".log")
    else:
        logger = rich

    logger.print(
        f"initial position noise {initialization_noise[0] + args_init_pos_noise / 1000}"
    )
    logger.print(
        f"initial orientation noise {initialization_noise[1] + args_init_ori_noise / 180 * np.pi}"
    )
    logger.print(
        f"hole pos estimation error {pose_estimation_error[0] + args_est_pos_error / 1000}"
    )
    logger.print(
        f"hole ori estimation error {pose_estimation_error[1] + args_est_ori_error / 180 * np.pi}"
    )

    # success rate
    success_list = []
    eps_time_list = []
    obs_pos_seq_all = []
    obs_quat_seq_all = []
    for i in range(args_eval_eps):
        logger.print(f"\n Episode {i+1}")
        obs = env.reset()
        print("error: {}".format(env.unwrapped.motion_generator.pos_error))
        logger.print(f"\n init pos noise {env.unwrapped.init_pos_noise*1000}")
        logger.print(
            f"\n init ori noise {env.unwrapped.init_rot_noise*180/np.pi}"
        )
        logger.print(
            f"\n hole pos error {env.unwrapped.pos_estimation_error*1000}"
        )
        logger.print(
            f"\n hole ori error {env.unwrapped.ori_estimation_error*180/np.pi}\n"
        )
        seq = []
        obs_pos_seq = []
        obs_quat_seq = []
        obs_pos_seq.append(list(env.motion_generator.eef_pos_task))
        obs_quat_seq.append(list(env.motion_generator.eef_quat_task))
        success = False
        steps = 0
        exec_time=0
        old_obs = obs.copy()
        # while not success and steps < 20:
        while not success and (exec_time < 30 and steps <100):
            # action = policy.get_action(obs)[0]
            action = policy.get_eval_action(obs)[0]
            # print(action, mp_description)
            print(obs)
            obs, rew, done, info = env.step(action)
            # print(obs - old_obs)
            old_obs = obs.copy()
            mp_description = get_short_description(
                env.get_mp_type(), env.get_mp_params()
            )
            mp_long_description = get_long_description(
                env.get_mp_type(), env.get_mp_params()
            )
            success = info["success"]
            steps += 1
            mp_status = info["mp_status"]
            mp_time = info["mp_time"]
            exec_time+=info["mp_time"]
            logger.print(
                f"{steps}: action idx {action}, {mp_long_description}, mp_status: {mp_status}, exec: {mp_time}"
            )
            seq.append(mp_description)
            obs_pos_seq.append(list(env.motion_generator.eef_pos_task))
            obs_quat_seq.append(list(env.motion_generator.eef_quat_task))
            print(f"current pos {env.motion_generator.eef_pos_task}")
            
            quat_error, axis_angle_error = T.quat_error(np.array(goal_quat), np.array(robot.eef_quat))
            rot_mat_error = T.quat2mat(quat_error)
            euler_error = T.mat2euler(rot_mat_error)
            roll_error = np.rad2deg(euler_error[0])
            pitch_error = np.rad2deg(euler_error[1])
            yaw_error = np.rad2deg(euler_error[2])
            if (roll_error > 90 or pitch_error > 90 or yaw_error > 90):
                break

        success_list.append(int(info["success"]))
        eps_time_list.append(info["eps_time"])
        obs_pos_seq_all.append(obs_pos_seq)
        obs_quat_seq_all.append(obs_quat_seq)
        logger.print(f"episode info {info}")
        logger.print(f"sequence {seq}")
        if success is True:
            break
            
    # reset
    env.reset()
    success_rate = np.sum(success_list) / args_eval_eps
    if success_rate == 0:
        eps_time_list = []
        avr_eps_time = 0
    else:
        eps_time_list = np.array(success_list) * np.array(eps_time_list)
        eps_time_list = eps_time_list[np.where(eps_time_list != 0)]
        avr_eps_time = np.mean(eps_time_list)

    # TODO average execution time
    logger.print(f"average execution time {avr_eps_time}")
    logger.print(f"std execution time {np.std(eps_time_list)}")
    
    # TODO max force
    logger.print(f"success_rate {success_rate}")
    # TODO save execution time to plot historgram
    if args_log:
        data = dict(
            episode_success=success_list,
            episode_time=list(eps_time_list),
            obs_pos_seq=obs_pos_seq_all,
            obs_quat_seq=obs_quat_seq_all,
        )
        data_file_name = "data_" + log_file_name + ".json"
        with open(os.path.join(logger.log_dir, data_file_name), "w") as f:
            json.dump(data, f)

    if success is True:
        return True, exec_time
    else:
        return False, exec_time

def get_short_description(type, params):
    if type == "InsertByAdmittance":
        return "insert {}N".format(params["target_force"][2])
    elif type == "MoveFixedDistance":
        direction_idx = np.where(params["direction"] != 0)[0][0]
        axes = ["x", "y", "z"]
        if direction_idx < 3:
            prefix = "t"
            ax = axes[direction_idx]
            target_distance = params["target_distance"] * 1000
        else:
            prefix = "r"
            ax = axes[direction_idx - 3]
            target_distance = params["target_distance"] * 180 / np.pi
        if params["direction"][direction_idx] < 0:
            ax = "-" + ax
        return "{}{}{:.2f}".format(prefix, ax, target_distance)
    elif type == "MoveToContact":
        direction_idx = np.where(params["direction"] != 0)[0][0]
        axes = ["x", "y", "z"]
        if direction_idx < 3:
            prefix = "tc"
            ax = axes[direction_idx]
        else:
            prefix = "rc"
            ax = axes[direction_idx - 3]
        if params["direction"][direction_idx] < 0:
            ax = "-" + ax
        return "{}{}".format(prefix, ax)


def get_long_description(type, params):
    if type == "InsertByAdmittance":
        if isinstance(params["target_force"], Iterable):
            target_force = params["target_force"][2]
        else:
            target_force = params["target_force"]
        return "insert with force {}N and admittance gain {}, max duration: {}s".format(
            target_force,
            params["admittance_gain"],
            params["max_duration"],
        )
    elif type == "MoveFixedDistance":
        direction_idx = np.where(params["direction"] != 0)[0][0]
        axes = ["x", "y", "z"]
        if direction_idx < 3:
            prefix = "translate"
            unit = "m"
            ax = axes[direction_idx]
        else:
            prefix = "rotate"
            unit = "rad"
            ax = axes[direction_idx - 3]
        if params["direction"][direction_idx] < 0:
            ax = "-" + ax
        return "{} {} {}{} with speed {}".format(
            prefix, ax, params["target_distance"], unit, params["speed"]
        )

    elif type == "MoveToContact":
        direction_idx = np.where(params["direction"] != 0)[0][0]
        axes = ["x", "y", "z"]
        if direction_idx < 3:
            prefix = "translate"
            ax = axes[direction_idx]
        else:
            prefix = "rotate"
            ax = axes[direction_idx - 3]
        if params["direction"][direction_idx] < 0:
            ax = "-" + ax

        return "{} {} until f > {}N with speed {}".format(
            prefix, ax, params["force_thresh"], params["speed"]
        )


# monitor
def run_monitor():
    monitor = MotionGeneratorMonitor()
    monitor.live_plot()
    import matplotlib.pyplot as plt

    plt.close("all")


if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description="evaluat policy on real robot")

    parser.add_argument(
        "-p",
        "--exp-path",
        type=str,
        help="path to the experiment folder containing the policy",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=1,
        help="path to additional config to initialize environment",
    )

    parser.add_argument(
        "-ip",
        "--init-pos-noise",
        type=float,
        default=0.0,
        help="for testing generalization, additional initialization noise (pos) in mm",
    )

    parser.add_argument(
        "-ir",
        "--init-ori-noise",
        type=float,
        default=0.0,
        help="for testing generalization, additional initialization noise (ori) in deg",
    )

    parser.add_argument(
        "-ep",
        "--est-pos-error",
        type=float,
        default=0.0,
        help="for testing generalization, additional hole pos estimation error in mm",
    )

    parser.add_argument(
        "-er",
        "--est-ori-error",
        type=float,
        default=0.0,
        help="for testing generalization, additional hole ori estimation error in deg",
    )

    parser.add_argument(
        "--eval-eps",
        type=int,
        default=1,
        help="number of eval episode for evaluation",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="whether to log the evaluation",
    )
    args = parser.parse_args()

    # run monitor
    monitor_process = mp.Process(target=run_monitor)
    monitor_process.start()

    evaluate_experiment(args)
    input("press enter to terminate")
    monitor_process.terminate()
