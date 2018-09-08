import operator

import time
import numpy as np
from osim.env import ProstheticsEnv


def penalty(observation):
    x_head_pelvis = observation['body_pos']['head'][0]-observation['body_pos']['pelvis'][0]
    accept_x1 = -0.25
    accept_x2 = 0.25
    pe = .0

    if x_head_pelvis < accept_x1:
        pe = -2.* (x_head_pelvis-accept_x1)
    elif x_head_pelvis < accept_x2:
        pe = 0.0
    else:
        pe = -2.* (accept_x2 - x_head_pelvis)

    return pe

def bonus(observation):

    if observation['body_pos']['head'][1] <= 1.546:
        return .0

    pelvis_v = observation['body_vel']['pelvis'][0]
    lv = observation['body_vel']['toes_l'][0]
    rv = observation['body_vel']['pros_foot_r'][0]
    return min(max(.0, max(lv, rv)-pelvis_v), 1.0)

def relative_dict_to_list(observation):
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


def main():
    env = ProstheticsEnv(False)

    action_dim = np.prod(np.array(env.action_space.shape))
    noise_process = np.zeros(action_dim)
    exploration_theta = .15
    exploration_mu = .0
    exploration_sigma = .2

    start_mmt = time.time()
    obs = env.reset(False)
    c = penalty(obs)
    obs = relative_dict_to_list(obs)
    print(len(obs))
    while True:
        pass
    for _ in range(50):
        act = env.action_space.sample()
        noise_process = exploration_theta*(exploration_mu - noise_process) + exploration_sigma*np.random.randn(action_dim)
        #print(noise_process)
        act += noise_process

        obs, rwd, done, info  = env.step(act, False)
        c = penalty(obs)
        b = bonus(obs)
        print(b)
        obs = relative_dict_to_list(obs)
        if done:
            obs = env.reset(False)
            c = penalty(obs)
            obs = relative_dict_to_list(obs)

    end_mmt = time.time()
    print("%f" % (end_mmt-start_mmt))


if __name__=="__main__":
    main()
