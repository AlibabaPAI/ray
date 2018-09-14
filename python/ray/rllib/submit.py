#!/usr/bin/ev python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle

import gym
import ray
from ray.rllib.agents.agent import get_agent_class
from ray.rllib.agents.dqn.common.wrappers import wrap_dqn
from ray.rllib.models import ModelCatalog

import opensim as osim
from osim.http.client import Client
from toy_opensim import relative_dict_to_list


EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=None, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    return parser


# register opensim environment
from osim.env import ProstheticsEnv
from ray.tune.registry import register_env

def env_creator(env_config):
    return ProstheticsEnv(False)
register_env("prosthetics", env_creator)


def run(args, parser):
    if not args.config:
        # Load configuration from file
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        with open(config_path) as f:
            args.config = json.load(f)

    if not args.env:
        if not args.config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = args.config.get("env")

    ray.init(redis_address="10.183.28.144:52059")
    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=args.config)
    agent.restore(args.checkpoint)
    policy = agent.local_evaluator.policy_map["default"]


    print("*************************************")
    print("*************************************")
    print("*************************************")
    print("Begin submiting")
    # CrowdAI environment
    remote_base = "http://grader.crowdai.org:1729"
    crowdai_token = "23133f3b85be506e07f3f781756e5f3e"
    client = Client(remote_base)
    obs = client.env_create(crowdai_token, env_id='ProstheticsEnv')
    obs = relative_dict_to_list(obs)

    while True:
        act = policy.sess.run(
            policy.output_actions,
            feed_dict={
                policy.cur_observations: [obs],
                policy.stochastic: False,
                policy.eps: .0})[0]
        [obs, reward, done, info] = client.env_step(act.tolist(), True)
        obs = relative_dict_to_list(obs)
        print(obs)
        if done:
            obs = client.env_reset()
            if not obs:
                break
            obs = relative_dict_to_list(obs)

    print("Complete interaction")
    client.submit()

    # repeat actions
    client = Client(remote_base)
    obs = client.env_create(crowdai_token, env_id='ProstheticsEnv')
    obs = relative_dict_to_list(obs)
    repeat_cnt = 0
    act = None

    while True:
        if repeat_cnt == 0:
            act = policy.sess.run(
                policy.output_actions,
                feed_dict={
                    policy.cur_observations: [obs],
                    policy.stochastic: False,
                    policy.eps: .0})[0]
        [obs, reward, done, info] = client.env_step(act.tolist(), True)
        repeat_cnt = (repeat_cnt + 1) % 4
        obs = relative_dict_to_list(obs)
        print(obs)
        if done:
            obs = client.env_reset()
            repeat_cnt = 0
            if not obs:
                break
            obs = relative_dict_to_list(obs)

    print("Complete interaction")
    client.submit()

    print("done.")


if __name__=="__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
