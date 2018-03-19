from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os

import numpy as np
import tensorflow as tf

import ray
from ray.rllib.ddpg.ddpg_replay_evaluator import DDPGReplayEvaluator
from ray.rllib.ddpg.ddpg_evaluator import DDPGEvaluator
from ray.rllib.optimizers import AsyncOptimizer, LocalMultiGPUOptimizer, \
    LocalSyncOptimizer
from ray.rllib.agent import Agent
from ray.tune.result import TrainingResult


DEFAULT_CONFIG = dict(
    # === Model ===
    # Whether to use dueling ddpg
    #dueling=True,
    # Whether to use double ddpg
    #double_q=True,
    # Hidden layer sizes of the policy networks
    actor_hiddens=[8, 8, 8],
    # Hidden layer sizes of the policy networks
    critic_hiddens=[8, 8, 8],
    # N-step Q learning
    n_step=1,
    # Config options to pass to the model constructor
    model={},
    # Discount factor for the MDP
    gamma=0.99,
    # Arguments to pass to the env creator
    env_config={},

    # === Exploration ===
    # Max num timesteps for annealing schedules. Exploration is annealed from
    # 1.0 to exploration_fraction over this number of timesteps scaled by
    # exploration_fraction
    schedule_max_timesteps=100000,
    # Number of env steps to optimize for before returning
    timesteps_per_iteration=2000,
    # Fraction of entire training period over which the exploration rate is
    # annealed
    exploration_fraction=0.25,
    # exploration hyper-parameters are
    # initial UO-noise scale
    initial_noise_scale=0.1,
    # theta
    exploration_theta=0.15,
    # sigma
    exploration_sigma=0.2,
    # How many steps of the model to sample before learning starts.
    learning_starts=1024,
    # Update the target network every `target_network_update_freq` steps.
    # or update the target network softly when this is zero
    target_network_update_freq=0,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    tau=0.01,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then each
    # worker will have a replay buffer of this size.
    buffer_size=50000,
    # If True prioritized replay buffer will be used.
    prioritized_replay=False,
    # Alpha parameter for prioritized replay buffer
    prioritized_replay_alpha=0.6,
    # Initial value of beta for prioritized replay buffer
    prioritized_replay_beta0=0.4,
    # Number of iterations over which beta will be annealed from initial
    # value to 1.0. If set to None equals to schedule_max_timesteps
    prioritized_replay_beta_iters=None,
    # Epsilon to add to the TD errors when updating priorities.
    prioritized_replay_eps=1e-6,

    # === Optimization ===
    # Learning rate for adam optimizer
    actor_lr=1e-3,
    critic_lr=1e-3,
    # Weights for L2 regularization
    actor_l2_reg=1e-6,
    critic_l2_reg=1e-6,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    sample_batch_size=1,
    # Size of a batched sampled from replay buffer for training. Note that if
    # async_updates is set, then each worker returns gradients for a batch of
    # this size.
    train_batch_size=1024,
    # SGD minibatch size. Note that this must be << train_batch_size. This
    # config has no effect if gradients_on_workres is True.
    sgd_batch_size=1024,
    # If not None, clip gradients during optimization at this value
    grad_norm_clipping=None,
    # Arguments to pass to the rllib optimizer
    optimizer={},

    # === Tensorflow ===
    # Arguments to pass to tensorflow
    tf_session_args={
        "device_count": {"CPU": 2},
        "log_device_placement": False,
        "allow_soft_placement": True,
        "inter_op_parallelism_threads": 1,
        "intra_op_parallelism_threads": 1,
    },

    # === Parallelism ===
    # Number of workers for collecting samples with. Note that the typical
    # setting is 1 unless your environment is particularly slow to sample.
    num_workers=1,
    # Whether to allocate GPUs for workers (if > 0).
    num_gpus_per_worker=1,
    # (Experimental) Whether to update the model asynchronously from
    # workers. In this mode, gradients will be computed on workers instead of
    # on the driver, and workers will each have their own replay buffer.
    async_updates=False,
    # (Experimental) Whether to use multiple GPUs for SGD optimization.
    # Note that this only helps performance if the SGD batch size is large.
    multi_gpu=False)


class DDPGAgent(Agent):
    _agent_name = "DDPG"
    _allow_unknown_subkeys = [
        "model", "optimizer", "tf_session_args", "env_config"]
    _default_config = DEFAULT_CONFIG

    def _init(self):
        if self.config["async_updates"]:
            self.local_evaluator = DDPGEvaluator(
                self.registry, self.env_creator, self.config, self.logdir)
            remote_cls = ray.remote(
                num_cpus=1, num_gpus=self.config["num_gpus_per_worker"])(
                DDPGReplayEvaluator)
            remote_config = dict(self.config, num_workers=1)
            # In async mode, we create N remote evaluators, each with their
            # own replay buffer (i.e. the replay buffer is sharded).
            self.remote_evaluators = [
                remote_cls.remote(
                    self.registry, self.env_creator, remote_config,
                    self.logdir)
                for _ in range(self.config["num_workers"])]
            optimizer_cls = AsyncOptimizer
        else:
            self.local_evaluator = DDPGReplayEvaluator(
                self.registry, self.env_creator, self.config, self.logdir)
            # No remote evaluators. If num_workers > 1, the DDPGReplayEvaluator
            # will internally create more workers for parallelism. This means
            # there is only one replay buffer regardless of num_workers.
            self.remote_evaluators = []
            if self.config["multi_gpu"]:
                optimizer_cls = LocalMultiGPUOptimizer
            else:
                optimizer_cls = LocalSyncOptimizer

        self.optimizer = optimizer_cls(
            self.config["optimizer"], self.local_evaluator,
            self.remote_evaluators)
        self.saver = tf.train.Saver(max_to_keep=None)

        self.global_timestep = 0
        self.last_target_update_ts = 0
        self.num_target_updates = 0

        # enable illustration of the computation graph in TensorBoard
        self.file_writer = tf.summary.FileWriter(self.logdir, self.local_evaluator.sess.graph)

    def _train(self):
        start_timestep = self.global_timestep

        while (self.global_timestep - start_timestep <
               self.config["timesteps_per_iteration"]):

            if self.global_timestep < self.config["learning_starts"]:
                self._populate_replay_buffer()
            else:
                self.optimizer.step()

            stats = self._update_global_stats()

            if self.global_timestep - self.last_target_update_ts > \
                    self.config["target_network_update_freq"]:
                self.local_evaluator.update_target()
                self.last_target_update_ts = self.global_timestep
                self.num_target_updates += 1

        mean_100ep_reward = 0.0
        mean_100ep_length = 0.0
        num_episodes = 0
        exploration = -1

        for s in stats:
            mean_100ep_reward += s["mean_100ep_reward"] / len(stats)
            mean_100ep_length += s["mean_100ep_length"] / len(stats)
            num_episodes += s["num_episodes"]
            exploration = s["exploration"]

        result = TrainingResult(
            episode_reward_mean=mean_100ep_reward,
            episode_len_mean=mean_100ep_length,
            episodes_total=num_episodes,
            timesteps_this_iter=self.global_timestep - start_timestep,
            info=dict({
                "exploration": exploration,
                "num_target_updates": self.num_target_updates,
            }, **self.optimizer.stats()))

        return result

    def _update_global_stats(self):
        if self.remote_evaluators:
            stats = ray.get([
                e.stats.remote() for e in self.remote_evaluators])
        else:
            stats = self.local_evaluator.stats()
            if not isinstance(stats, list):
                stats = [stats]
        new_timestep = sum(s["local_timestep"] for s in stats)
        assert new_timestep > self.global_timestep, new_timestep
        self.global_timestep = new_timestep
        self.local_evaluator.set_global_timestep(self.global_timestep)
        for e in self.remote_evaluators:
            e.set_global_timestep.remote(self.global_timestep)
        return stats

    def _populate_replay_buffer(self):
        if self.remote_evaluators:
            for e in self.remote_evaluators:
                e.sample.remote(no_replay=True)
        else:
            self.local_evaluator.sample(no_replay=True)

    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for ev in self.remote_evaluators:
            ev.__ray_terminate__.remote(ev._ray_actor_id.id())

    def _save(self, checkpoint_dir):
        checkpoint_path = self.saver.save(
            self.local_evaluator.sess,
            os.path.join(checkpoint_dir, "checkpoint"),
            global_step=self.iteration)
        extra_data = [
            self.local_evaluator.save(),
            ray.get([e.save.remote() for e in self.remote_evaluators]),
            self.global_timestep,
            self.num_target_updates,
            self.last_target_update_ts]
        pickle.dump(extra_data, open(checkpoint_path + ".extra_data", "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.saver.restore(self.local_evaluator.sess, checkpoint_path)
        extra_data = pickle.load(open(checkpoint_path + ".extra_data", "rb"))
        self.local_evaluator.restore(extra_data[0])
        ray.get([
            e.restore.remote(d) for (d, e)
            in zip(extra_data[1], self.remote_evaluators)])
        self.global_timestep = extra_data[2]
        self.num_target_updates = extra_data[3]
        self.last_target_update_ts = extra_data[4]

    def compute_action(self, observation):
        return self.local_evaluator.ddpg_graph.act(
            self.local_evaluator.sess, np.array(observation)[None], False, 0.0)[0]