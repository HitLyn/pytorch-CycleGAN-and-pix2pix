import os
import h5py
import argparse
import random
import numpy as np

from os.path import join as pjoin
from PIL import Image

import robosuite
import random

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='/home/jonathan/Desktop/cs236/pytorch-CycleGAN-and-pix2pix/datasets', help='the base path of where to store the images')
parser.add_argument('--dataset_name', type=str, default='sawyer_traj', help='name of the dataset to create')
parser.add_argument('--phase', type=str, default='A', help='what phase of data we are creating (ie train, test, valid)')
parser.add_argument('--env', type=str, default='SawyerLift')
parser.add_argument('--n_train', type=int, deault=4000)
parser.add_argument('--n_test', type=int, default=400)
parser.add_argument('--n_val', type=int, default=400)

VALID_PHASES = ['trainA', 'trainB', 'testA', 'testB', 'valA', 'valB', 'A', 'B']
BASE_PATH = "/home/jonathan/Desktop/cs236/pytorch-CycleGAN-and-pix2pix/datasets/sawyer_traj/test_A"

if __name__ == "__main__":
    args = parser.parse_args()

    assert args.phase in VALID_PHASES, 'Phase must be one of the following: {}'.format(VALID_PHASES)

    if args.phase == 'A':
        to_create = ['trainA', 'testA', 'valA']
    elif args.phase == 'B':
        to_create = ['trainB', 'testB', 'valB']
    else:
        to_create = [args.phase]

    base_path = args.base_path + '/' + args.dataset_name + '/'

    env = robosuite.make(
        args.env,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=100,
        camera_name='agentview'
    )

    for phase in to_create:

        if 'train' in phase:
            n = args.n_train
        elif 'test' in phase:
            n = args.n_test
        else:
            n = args.n_val

        path = base_path + args.phase

        if not os.path.isdir(path):
            os.makedirs(path)

        env.reset()

        for frame_id in range(n):

            env.step(np.random.randn(env.dof))
            im = env._get_observation()["image"][::-1]
            im = Image.fromarray(im)
            path = pjoin(BASE_PATH, "img%06d.png"%frame_id)
            im.save(path)

            if bool(random.getrandbits(1)):
                env.reset()
