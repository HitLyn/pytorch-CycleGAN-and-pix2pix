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
import robosuite as suite

import pickle
import os
import time

import torch
import matplotlib.pyplot as plt
from os.path import join as pjoin
from glob import glob

from surreal.env import *
from surreal.main.ppo_configs import *
# from surreal.tmux.surreal_tmux import load_config
# from surreal.learner import learner_factory
from surreal.agent import PPOAgent
import surreal.utils as U
import cv2
import time

USER = "amandlek"
EXPERIMENT_NAME = "ppo-pegs-round-sparse-eplen-100-1000-1"
DESTINATION = "/Users/ajaymandlekar/Desktop/surreal_policy_files"
CONFIG_PATH = "ppo_configs.py"

def get_policy_params(user, experiment_name, destination):
    """
    Retrieves policy parameters from the surreal fs, if directory
    @destination/@experiment_name does not already exist.

    Unfortunately, must have an experiment with a running learner
    for this to work. 

    :param user: username
    :param experiment_name: name of the experiment, without the user string
    :param destination: place to put the policy parameters.

    :return ckpt_path: path to the saved checkpoint file.
    """
    folder = pjoin(destination, experiment_name)
    try:
        os.mkdir(folder)
        success = True
    except:
        print("Not retrieving parameters. Directory already exists, or other error encountered.")
        success = False

    if success:
        cmd = "kurreal scp learner:/fs/experiments/{}/{}/checkpoint {}".format(user, experiment_name, folder)
        os.system(cmd)

    # choose latest ckpt
    max_iter = -1.
    max_ckpt = None
    for ckpt in glob(pjoin(folder, "*.ckpt")):
        iter_num = int(ckpt.split('.')[1])
        if iter_num > max_iter:
            max_iter = iter_num
            max_ckpt = ckpt
    return max_ckpt

def restore_model(path_to_ckpt):
    """
    Loads model from a ckpt file.
    """
    with open(path_to_ckpt, 'rb') as fp:
        data = pickle.load(fp)
    return data['model']

def restore_config(path_to_config):
    """
    Loads a config from a file.
    """

    ### TODO: how to load config from YAML? ###

    # hard code additional args for now, but make this part better eventually... 
    additional_args = ['--env', 'robosuite:SawyerLift',
                       '--num-agents', '2',
                       '--num-gpus', '0',
                       '--agent-num-gpus', '0']
    configs = load_config(path_to_config, additional_args)
    return configs

def restore_env(env_config):
    """
    Restores the environment.
    """
    env_config.eval_mode.render = True
    env, env_config = make_env(env_config, 'eval')
    return env, env_config

def restore_agent(learner_config, env_config, session_config, model):
    """
    Restores an agent from a model.
    """
    learner_config.algo.use_z_filter = False
    agent_class = PPOAgent
    agent = agent_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=0,
        agent_mode='eval_deterministic_local',
    )
    agent.model.load_state_dict(model)
    return agent

if __name__ == "__main__":

    # set a seed
    # np.random.seed(int(time.time() * 100000 % 100000))

    # retrieve the policy params if necessary and restore the model
    path_to_policy_params = "/home/jonathan/Downloads/learner.166000.ckpt.cpu"
    print("\nLoading policy located at {}\n".format(path_to_policy_params))
    model = restore_model(path_to_policy_params)

    # restore the configs
    # configs = restore_config(CONFIG_PATH)
    session_config, learner_config, env_config = PPO_DEFAULT_SESSION_CONFIG, PPO_DEFAULT_LEARNER_CONFIG, PPO_DEFAULT_ENV_CONFIG

    session_config.agent.num_gpus = 0
    session_config.learner.num_gpus = 0
    # env_config.env_name = 'mujocomanip:SawyerPegsRoundEnv'

    # restore the environment
    env, env_config = restore_env(env_config)
    # restore the agent
    agent = restore_agent(learner_config, env_config, session_config, model)
    print("Successfully loaded agent and model!")
    reward = 0
    # do some rollouts
    opt = TransferOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.gpu_ids = []
    model = create_model(opt)
    model.setup(opt)

    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    transform = data.base_dataset.get_transform(opt)

    env.unwrapped.camera_width = 256
    env.unwrapped.camera_height = 256
    
    for j in range(20) :
        ob, info = env.reset()
        # env.unwrapped.viewer.viewer._hide_overlay = True
        # env.unwrapped.viewer.set_camera(2)
        for i in range(200):
            obs = env.unwrapped._get_observation()["image"][::-1]

            obs = Image.fromarray(obs).convert('RGB')
            #obs = obs.resize((256,256))
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)

            plt.imshow(obs)
            obs = transform(obs)
            obs = obs.unsqueeze(0)


            ax.set_title('original')
            ax = fig.add_subplot(1,2,2)
            transfered = model.inference(opt.direction, obs)
            transfered = transfered[0].cpu().detach().numpy()
            transfered = (np.transpose(transfered, (1, 2, 0)) + 1) / 2.0 * 255.0
            transfered = transfered.astype(np.uint8)

            plt.imshow(transfered)
            ax.set_title('transfered')

            plt.show(block=False)
            #plt.waitforbuttonpress()
            time.sleep(1)
            plt.close()
            transfered = Image.fromarray(transfered).convert('RGB')
            transfered = transfered.resize((84,84))
            transfered = np.transpose(np.array(transfered),(2,0,1))
            transferred =transfered[::-1]
            ob['pixel']['camera0']= transfered[::-1]
            a = agent.act(ob)
            ob, r, _, _ = env.step(a)
            print(r)
            # plt.imsave('trained/'+str(j)+'-'+str(i)+'.png',obs['image'],origin='lower')
            reward +=r
            # NOTE: we need to unwrap the environment here because some wrappers override render
            # env.unwrapped.render()
    print(reward/10)
