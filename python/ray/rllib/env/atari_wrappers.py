import operator
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


def is_atari(env):
    return hasattr(env, "unwrapped") and hasattr(env.unwrapped, "ale")


def get_wrapper_by_cls(env, cls):
    """Returns the gym env wrapper of the given class, or None."""
    currentenv = env
    while True:
        if isinstance(currentenv, cls):
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            return None


class MonitorEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Record episodes stats prior to EpisodicLifeEnv, etc."""
        gym.Wrapper.__init__(self, env)
        self._current_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._num_episodes = 0
        self._num_returned = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        self._current_reward = 0
        self._num_steps = 0

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        return (obs, rew, done, info)

    def get_episode_rewards(self):
        return self._episode_rewards

    def get_episode_lengths(self):
        return self._episode_lengths

    def get_total_steps(self):
        return self._total_steps

    def next_episode_results(self):
        for i in range(self._num_returned, len(self._episode_rewards)):
            yield (self._episode_rewards[i], self._episode_lengths[i])
        self._num_returned = len(self._episode_rewards)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset.

        For environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2, ) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, dim):
        """Warp frames to the specified size (dim x dim)."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


def wrap_deepmind(env, dim=84, framestack=True):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if 'NoFrameskip' in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, dim)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        env = FrameStack(env, 4)
    return env


class WalkingEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        add 1 to original reward for each timestep except for the terminal one
        repeat an action for 4 timesteps
        """
        gym.Wrapper.__init__(self, env)
        self.observation_space.shape = (306,)
        self._skip = skip

    def _penalty(self, observation):
        x_head_pelvis = observation['body_pos']['head'][0]-observation['body_pos']['pelvis'][0]

        # height from pelvis to head is around 0.62
        # consider 0.62 * cos(60) first
        # consider 0.62 / sqrt(2) later
        accept_x1 = -0.31
        accept_x2 = 0.31
        if x_head_pelvis < accept_x1:
            pe = 5.0
            done = True
        elif x_head_pelvis < accept_x2:
            pe = 0.0
            done = False
        else:
            pe = 1.0
            done = False

        z_head_pelvis = observation['body_pos']['head'][2]-observation['body_pos']['pelvis'][2]
        accept_z1 = -0.31
        accept_z2 = 0.31
        if z_head_pelvis < accept_z1:
            pe += 5.0
            done = True
        elif z_head_pelvis < accept_z2:
            pass
        else:
            pe += 5.0
            done = True

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
            'body_acc': list(map(lambda v: v/100.0, observation['body_acc']['pelvis']))
        }

        res += pelvs['body_pos']
        res += pelvs['body_vel']
        res += pelvs['body_acc']

        # Body Observations
        for info_type in ['body_pos', 'body_vel']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += list(map(operator.sub, observation[info_type][body_part], pelvs[info_type]))

        for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                          'femur_l', 'femur_r', 'head',
                          'torso', 'pros_foot_r', 'pros_tibia_r']:
            res += list(map(lambda a,b: a/100.0-b, observation[info_type][body_part], pelvs['body_acc']))

        for info_type in ['body_pos_rot', 'body_vel_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += observation[info_type][body_part]

        for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                          'femur_l', 'femur_r', 'head', 'pelvis',
                          'torso', 'pros_foot_r', 'pros_tibia_r']:
            res += list(map(lambda v: v/1000.0, observation['body_acc_rot'][body_part]))

        # ground_pelvis
        res += list(map(operator.sub, observation['joint_pos']['ground_pelvis'][0:3], pelvs['body_pos']))
        res += observation['joint_pos']['ground_pelvis'][3:6]
        res += list(map(operator.sub, observation['joint_vel']['ground_pelvis'][0:3], pelvs['body_vel']))
        res += observation['joint_vel']['ground_pelvis'][3:6]
        res += list(map(lambda a,b: a/100.0-b, observation['joint_acc']['ground_pelvis'][0:3], pelvs['body_acc']))
        res += list(map(lambda v: v/1000.0, observation['joint_acc']['ground_pelvis'][3:6]))

        # joint
        for info_type in ['joint_pos', 'joint_vel']:
            for joint in ['ankle_l', 'ankle_r', 'back',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += observation[info_type][joint]

        for joint in ['ankle_l', 'ankle_r', 'back',
                      'hip_l', 'hip_r', 'knee_l', 'knee_r']:
            res += list(map(lambda v: v/1000.0, observation['joint_acc'][joint]))

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
