#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

import ray
from ray.tune.config_parser import make_parser, resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    rllib train --run DQN --env CartPole-v0

Grid search example via RLlib CLI:
    rllib train -f tuned_examples/cartpole-grid-search-example.yaml

Grid search example via executable:
    ./train.py -f tuned_examples/cartpole-grid-search-example.yaml

Note that -f overrides all other trial-specific command-line options.
"""


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--redis-address",
        default=None,
        type=str,
        help="The Redis address of the cluster.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to pass to Ray."
        " This only has an affect in local mode.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to pass to Ray."
        " This only has an affect in local mode.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "--env", default=None, type=str, help="The gym environment to use.")
    parser.add_argument(
        "--queue-trials",
        action='store_true',
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    return parser


# wrapper required
import operator
import numpy as np
from collections import deque
import gym
from gym import spaces

# register opensim environment
from osim.env import ProstheticsEnv
from ray.tune.registry import register_env

class WalkingEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        add 1 to original reward for each timestep except for the terminal one
        repeat an action for 4 timesteps
        """
        gym.Wrapper.__init__(self, env)
        self.observation_space.shape = (223,)
        self._skip = skip

    def _penalty(self, observation):
        x_head_pelvis = observation['body_pos']['head'][0]-observation['body_pos']['pelvis'][0]

        # height from pelvis to head is around 0.62
        # consider 0.62 * cos(60) first
        # consider 0.62 / sqrt(2) later
        accept_x1 = -0.31
        accept_x2 = 0.31
        if x_head_pelvis < accept_x1:
            pe = .667
            #pe = 5.0
            #done = True
        elif x_head_pelvis < accept_x2:
            pe = 0.0
            #done = False
        else:
            pe = 0.667
            #done = False

        z_head_pelvis = observation['body_pos']['head'][2]-observation['body_pos']['pelvis'][2]
        accept_z1 = -0.31
        accept_z2 = 0.31
        if z_head_pelvis < accept_z1:
            pe += 0.667
            #pe += 5.0
            #done = True
        elif z_head_pelvis < accept_z2:
            pass
        else:
            pe += 0.667
            #pe += 5.0
            #done = True

        pelvis_pos_y = observation["body_pos_rot"]["pelvis"][1]
        accept_y1 = -0.52
        accept_y2 = 0.52
        if pelvis_pos_y < accept_y1:
            pe += 2.0*(accept_y1-pelvis_pos_y)
        elif pelvis_pos_y < accept_y2:
            pass
        else:
            pe += 2.0*(pelvis_pos_y-accept_y2)

        done = observation['body_pos']['pelvis'][1] <= 0.65

        return pe, done

    def _bonus(self, observation):
        pelvis_v = observation['body_vel']['pelvis'][0]
        lv = observation['body_vel']['toes_l'][0]
        rv = observation['body_vel']['pros_foot_r'][0]
        return min(max(.0, max(lv, rv)-pelvis_v), 1.0)

    def _relative_dict_to_list(self, observation):
        res = []

        pelvs = {
            'body_pos': observation['body_pos']['pelvis'],
            'body_vel': observation['body_vel']['pelvis'],
            #'body_acc': list(map(lambda v: v/100.0, observation['body_acc']['pelvis']))
        }

        res += pelvs['body_pos']
        res += pelvs['body_vel']
        #res += pelvs['body_acc']

        # Body Observations
        for info_type in ['body_pos', 'body_vel']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += list(map(operator.sub, observation[info_type][body_part], pelvs[info_type]))

        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda a,b: a/100.0-b, observation['body_acc'][body_part], pelvs['body_acc']))

        for info_type in ['body_pos_rot', 'body_vel_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += observation[info_type][body_part]
                #if body_part == "pelvis":
                #    print(observation[info_type][body_part])
        #print("***********************************************************************************")

        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head', 'pelvis',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda v: v/1000.0, observation['body_acc_rot'][body_part]))

        # ground_pelvis
        res += list(map(operator.sub, observation['joint_pos']['ground_pelvis'][0:3], pelvs['body_pos']))
        res += observation['joint_pos']['ground_pelvis'][3:6]
        res += list(map(operator.sub, observation['joint_vel']['ground_pelvis'][0:3], pelvs['body_vel']))
        res += observation['joint_vel']['ground_pelvis'][3:6]
        #res += list(map(lambda a,b: a/100.0-b, observation['joint_acc']['ground_pelvis'][0:3], pelvs['body_acc']))
        #res += list(map(lambda v: v/1000.0, observation['joint_acc']['ground_pelvis'][3:6]))

        # joint
        for info_type in ['joint_pos', 'joint_vel']:
            for joint in ['ankle_l', 'ankle_r', 'back',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += observation[info_type][joint]

        #for joint in ['ankle_l', 'ankle_r', 'back',
        #              'hip_l', 'hip_r', 'knee_l', 'knee_r']:
        #    res += list(map(lambda v: v/1000.0, observation['joint_acc'][joint]))

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r',
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(observation['muscles'][muscle]['activation'])
            #res.append(observation['muscles'][muscle]['fiber_force']/5000.0)
            res.append(observation['muscles'][muscle]['fiber_length'])
            res.append(observation['muscles'][muscle]['fiber_velocity'])

        return res
    
    def step(self, ac):
        total_reward = .0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(ac, False)
            penalty, strong_done = self._penalty(obs)
            b = self._bonus(obs)
            done = done if done else strong_done
            total_reward += (reward if done else reward+1.0) - penalty + b
            if done:
                break

        return self._relative_dict_to_list(obs), total_reward, done, info

    def reset(self, **kwargs):
        return self._relative_dict_to_list(self.env.reset(project=False, **kwargs))

def wrap_opensim(env):
    env = WalkingEnv(env)
    return env
def env_creator(env_config):
    env = ProstheticsEnv(False)
    return wrap_opensim(env)
from ray.tune.registry import register_env
register_env("prosthetics", env_creator)


def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.load(f)
    else:
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "local_dir": args.local_dir,
                "trial_resources": (
                    args.trial_resources and
                    resources_to_json(args.trial_resources)),
                "stop": args.stop,
                "config": dict(args.config, env=args.env),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    for exp in experiments.values():
        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")

    ray.init(
        redis_address=args.redis_address,
        num_cpus=args.ray_num_cpus,
        num_gpus=args.ray_num_gpus)
    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
