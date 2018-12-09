from options.transfer_options import TransferOptions
from models import create_model
import numpy as np
import data
from PIL import Image

import matplotlib.pyplot as plt

import pickle

from surreal.env import *
from surreal.main.ppo_configs import *
from surreal.agent import PPOAgent


def restore_model(path_to_ckpt):
    """
    Loads model from a ckpt file.
    """
    with open(path_to_ckpt, 'rb') as fp:
        data = pickle.load(fp)
    return data['model']

def restore_env(env_config, internal):
    """
    Restores the environment.
    """
    env_config.eval_mode.render = True
    env, env_config = make_env(env_config, 'eval', internal=internal)
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

    opt = TransferOptions().parse()

    # retrieve the policy params if necessary and restore the model
    print("\nLoading policy located at {}\n".format(opt.policy_path))
    model = restore_model(opt.policy_path)

    # restore the configs
    session_config, learner_config, env_config = PPO_DEFAULT_SESSION_CONFIG, PPO_DEFAULT_LEARNER_CONFIG, PPO_DEFAULT_ENV_CONFIG

    session_config.agent.num_gpus = 0
    session_config.learner.num_gpus = 0
    
    # restore the environment
    env, env_config = restore_env(env_config, internal=True)

    # restore the agent
    agent = restore_agent(learner_config, env_config, session_config, model)
    print("Successfully loaded agent and model!")
    reward = 0

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

    env.unwrapped.camera_width, env.unwrapped.camera_height = opt.fineSize, opt.fineSize

    if opt.save_obs:
        os.makedirs(opt.obs_save_path, exist_ok=True)

    count = 0
    ep_rewards = []

    fig = plt.figure()

    for j in range(10) :
        ob, info = env.reset()

        for i in range(200):

            obs = env.unwrapped._get_observation()["image"]
            obs = Image.fromarray(obs[::-1]).convert('RGB')

            if opt.save_obs:
                obs.save(opt.obs_save_path + '/rollout_{}_{}.jpg'.format(j, i))
            else:
                plt.subplot(121)
                plt.imshow(obs)

            obs = transform(obs).unsqueeze(0)

            transfered = model.inference(opt.direction, obs)
            transfered = transfered[0].cpu().detach().numpy()
            transfered = (np.transpose(transfered, (1, 2, 0)) + 1) / 2.0 * 255.0
            transfered = transfered.astype(np.uint8)

            transfered = Image.fromarray(transfered[::-1]).convert('RGB')
            if not opt.save_obs:
                plt.subplot(122)
                plt.imshow(np.array(transfered)[::-1])
                plt.pause(0.01)
                plt.draw()
            transfered = transfered.resize((84,84))

            transfered = np.transpose(transfered,(2,0,1)).copy()

            ob['pixel']['camera0']= transfered
            a = agent.act(ob)
            ob, r, done, _ = env.step(a)
            #if done:
            #   env.reset()
 
            reward +=r
            count += 1
        ep_rewards.append(reward - np.sum(ep_rewards)) if len(ep_rewards) != 0 else ep_rewards.append(reward)
        print(np.sum(ep_rewards) / len(ep_rewards))

    print(reward/10)
