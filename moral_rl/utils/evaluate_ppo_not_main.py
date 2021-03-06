from tqdm import tqdm
from moral.ppo import *
import torch
from envs.gym_wrapper import *
import numpy as np


def evaluate_ppo(ppo, config, n_eval=1000):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards
    """
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    obj_logs = []
    obj_returns = []

    actions_chosen = np.zeros(10)
    # previous_infos = 0

    for t in range(n_eval):
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = env.step(actions)
        obj_logs.append(reward)


        actions_chosen[actions] += 1
        # if actions == 4:
        #     # print("states_tensor = ", states_tensor.shape)
        #     # print("next_states = ", next_states.shape)
        #     print("reward = ", reward)
        #     print("info = ", info)
        #     print("previous_infos = ", previous_infos)
        # previous_infos = info

        if done:
            next_states = env.reset()
            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            obj_logs = []

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    obj_returns = np.array(obj_returns)
    obj_means = obj_returns.mean(axis=0)
    obj_std = obj_returns.std(axis=0)

    print("action chosen = ", actions_chosen)

    return list(obj_means), list(obj_std)


def evaluate_ppo_discrim(ppo, discrim, config, n_eval=1000):
    """
    :param ppo: Trained policy
    :param config: Environment config
    :param n_eval: Number of evaluation steps
    :return: mean, std of rewards, mean std of discrim evaluation of actions
    """
    env = GymWrapper(config.env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    obj_logs = []
    obj_returns = []
    discrim_logs = []
    discrim_returns = []

    for t in range(n_eval):
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = env.step(actions)
        obj_logs.append(reward)
        print("states = ", states)
        print("len states = ", len(states))
        discrim_logs.append(discrim.forward(states, next_states, config.gamma))

        
        # for i in len(states):
        #     discrim_logs.append(discrim.forward(states[i], next_states[i], config.gamma))

        if done:
            next_states = env.reset()
            obj_logs = np.array(obj_logs).sum(axis=0)
            obj_returns.append(obj_logs)
            obj_logs = []
            discrim_logs = np.array(discrim_logs).sum(axis=0)
            discrim_returns.append(discrim_logs)
            discrim_logs = []

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    obj_returns = np.array(obj_returns)
    obj_means = obj_returns.mean(axis=0)
    obj_std = obj_returns.std(axis=0)

    discrim_returns = np.array(discrim_returns)
    discrim_means = discrim_returns.mean(axis=0)
    discrim_std = discrim_returns.std(axis=0)

    return list(obj_means), list(obj_std), list(discrim_means), list(discrim_std)