import os
from surreal.env import *
from surreal.main.ppo_configs import *
from PIL import Image
import matplotlib.pyplot as plt
from options.datagen_options import DataGenOptions
from mujoco_py import MjSimState
import pickle

def restore_env(env_config, size, textured=False, collision=False):
    """
    Restore the env
    """
    env_config.eval_mode.render = False
    env, env_config = make_env(env_config, 'eval')

    env.unwrapped.camera_width, env.unwrapped.camera_height = size, size
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

def restore_model(path_to_ckpt):
    """
    Loads model from a ckpt file.
    """
    with open(path_to_ckpt, 'rb') as fp:
        data = pickle.load(fp)
    return data['model']


def rollout_save_states(opt, env, env_config):
    # retrieve the policy params if necessary and restore the model
    print("\nLoading policy located at {}\n".format(opt.policy_path))
    model = restore_model(opt.policy_path)

    env.unwrapped.camera_width, env.unwrapped.camera_height = 84, 84

    # restore the configs
    session_config, learner_config = PPO_DEFAULT_SESSION_CONFIG, PPO_DEFAULT_LEARNER_CONFIG

    session_config.agent.num_gpus = 0
    session_config.learner.num_gpus = 0

    agent = restore_agent(learner_config, env_config, session_config, model)
    print("Successfully loaded agent and model!")

    if not opt.save_obs:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    states = []
    for rollout in range(opt.n_rollouts):
        if opt.textured:
            with open('mujoco/model.xml', 'r') as f:
                xml = f.read()
            d = os.path.dirname(os.path.abspath(__file__))
            env.unwrapped.reset_from_xml_string(xml.format(base=d))
            ob = env.unwrapped._get_observation()
            ob = env._flatten_obs(ob)
        else:
            ob, _ = env.reset()


        for step in range(opt.n_steps):
            obs = env.unwrapped._get_observation()["image"]

            """
            plt.imshow(obs[::-1])
            ax.set_title('Observation')
            plt.pause(0.01)
            plt.draw()
            """

            state = env.unwrapped.sim.get_state().flatten().copy()
            states.append(state)

            ob['pixel']['camera0'] = np.transpose(obs, (2,0,1))


            action = agent.act(ob)
            ob, _, _, _ = env.step(action)

    print()
    np.save(opt.states_file, np.array(states))

def rollout_from_state(opt, env):

    t = 'textured' if opt.textured or opt.collision else 'normal'
    os.makedirs(opt.data_root + '/' + opt.dataset_name + '/state_rollout_{}'.format(t), exist_ok=True)

    save_path = opt.data_root + '/' + opt.dataset_name + '/state_rollout_{}'.format(t)

    states = np.load(opt.states_file + '.npy')

    if opt.textured:
        with open('mujoco/model.xml', 'r') as f:
            xml = f.read()
        d = os.path.dirname(os.path.abspath(__file__))
        env.unwrapped.reset_from_xml_string(xml.format(base=d))
    else:
        env.reset()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for state_index in range(states.shape[0]):
        if opt.noisy:
            states[state_index] += np.random.randn(states[state_index].shape)

        state = MjSimState.from_flattened(states[state_index], env.unwrapped.sim)

        env.unwrapped.sim.set_state(state)
        env.unwrapped.sim.forward()
        obs = env.unwrapped._get_observation()['image'][::-1]
        obs = Image.fromarray(obs).convert('RGB')

        obs.save(save_path + '/from_states_{}.jpg'.format(state_index))

        """
        plt.imshow(obs)
        ax.set_title('Observation')
        plt.pause(0.01)
        plt.draw()
        """

def rollout_policy(opt, env, env_config):

    os.makedirs(opt.data_root + '/' + opt.dataset_name + '/rollout', exist_ok=True)

    save_path = opt.data_root + '/' + opt.dataset_name + '/rollout'
    # retrieve the policy params if necessary and restore the model
    print("\nLoading policy located at {}\n".format(opt.policy_path))
    model = restore_model(opt.policy_path)

    env.unwrapped.camera_width, env.unwrapped.camera_height = 84, 84

    # restore the configs
    session_config, learner_config = PPO_DEFAULT_SESSION_CONFIG, PPO_DEFAULT_LEARNER_CONFIG

    session_config.agent.num_gpus = 0
    session_config.learner.num_gpus = 0

    agent = restore_agent(learner_config, env_config, session_config, model)
    print("Successfully loaded agent and model!")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    total_reward = 0
    for rollout in range(opt.n_rollouts):
        if opt.textured:
            with open('mujoco/model.xml', 'r') as f:
                xml = f.read()
            d = os.path.dirname(os.path.abspath(__file__))
            env.unwrapped.reset_from_xml_string(xml.format(base=d))
            ob = env.unwrapped._get_observation()
            ob = env._flatten_obs(ob)
        else:
            ob, _ = env.reset()

        for step in range(opt.n_steps):
            obs = env.unwrapped._get_observation()["image"]

            ob['pixel']['camera0'] = np.transpose(obs, (2,0,1))
            action = agent.act(ob)
            ob, r, done, _ = env.step(action)

            total_reward += r

            o = Image.fromarray(obs[::-1]).resize((opt.size, opt.size))
            o.save(save_path + '/rollout_{}_{}.jpg'.format(rollout, step))

            """
            plt.imshow(obs)
            ax.set_title('Observation')
            plt.pause(0.01)
            plt.draw()
            """


    print('Average return:   {}'.format(total_reward / (opt.n_rollouts * opt.n_steps)))

def rollout_random(opt, env):

    t = 'textured' if opt.textured or opt.collision else 'normal'
    if opt.save_obs:
        os.makedirs(opt.data_root + '/' + opt.dataset_name + '/random_{}'.format(t), exist_ok=True)

    save_path = opt.data_root + '/' + opt.dataset_name + '/random_{}'.format(t)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for rollout in range(opt.n_rollouts):
        if opt.textured:
            with open('mujoco/model.xml', 'r') as f:
                xml = f.read()
            d = os.path.dirname(os.path.abspath(__file__))
            env.unwrapped.reset_from_xml_string(xml.format(base=d))
        else:
            env.reset()
        for step in range(opt.n_steps):
            obs = env.unwrapped._get_observation()['image'][::-1]
            obs = Image.fromarray(obs).convert('RGB')

            obs.save(save_path + '/random_{}_{}.jpg'.format(rollout, step))

            """
            plt.imshow(obs)
            ax.set_title('Observation')
            plt.pause(0.01)
            plt.draw()
            """

            action = np.random.randn(env.unwrapped.dof)
            env.step(action)


if __name__ == '__main__':
    options = DataGenOptions().parse()

    environment_config = PPO_DEFAULT_ENV_CONFIG
    environment, _ = restore_env(environment_config, options.size,
                                 textured=options.textured, collision=options.collision)


    if options.mode == 'save_states':
        rollout_save_states(options, environment, environment_config)
    elif options.mode == 'state2im':
        rollout_from_state(options, environment)
    elif options.mode == 'rollout':
        rollout_policy(options, environment, environment_config)
    elif options.mode == 'random':
        rollout_random(options, environment)
