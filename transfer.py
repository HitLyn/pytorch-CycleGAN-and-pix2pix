import os
from options.transfer_options import TransferOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import data
from PIL import Image

import matplotlib.pyplot as plt

#import robosuite
import RoboticsSuite as suite

if __name__ == '__main__':
    opt = TransferOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display

    model = create_model(opt)
    model.setup(opt)

    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    env = suite.make(opt.env,
                     has_renderer=False,
                     ignore_done=True,
                     use_camera_obs=True,
                     #gripper_visualization=True,
                     reward_shaping=True,
                     control_freq=100,
                     camera_name='agentview'
    )
    transform = data.base_dataset.get_transform(opt)
    for ep in range(opt.num_test):
        print('Starting ep {}'.format(ep))
        env.reset()

        while not env.done:
            # get action
            env.step(np.random.randn(env.dof))
            obs = env._get_observation()["image"][::-1]

            obs_save = obs.copy()

            obs = Image.fromarray(obs).convert('RGB')
            obs = transform(obs)
            obs = obs.unsqueeze(0)
            transfered = model.inference(opt.direction, obs)

            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)

            plt.imshow(obs_save)
            ax.set_title('original')
            ax = fig.add_subplot(1,2,2)
            transfered = transfered[0].cpu().detach().numpy()
            transfered = (np.transpose(transfered, (1, 2, 0)) + 1) / 2.0 * 255.0
            transfered = transfered.astype(np.uint8)

            plt.imshow(transfered)
            ax.set_title('transfered')

            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close()
