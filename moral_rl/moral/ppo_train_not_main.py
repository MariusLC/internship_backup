from moral.ppo import *
from envs.gym_wrapper import *
from utils.save_data import *

import torch
from tqdm import tqdm
import wandb
import argparse


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# OLD
# def ppo_train_n_experts(nb_experts, env, env_rad, lambd_list, lambd_str_list, ppo_filenames, model_path, model_ext):
#     for i in range(nb_experts):
#         filename = model_path+ppo_filenames+env+lambd_str_list[i]+model_ext
#         ppo_train_1_expert(env_rad+env, lambd_list[i], filename)

# NEW
def ppo_train_n_experts(env, env_steps_ppo, lambd_list, experts_filenames):
    for i in range(len(lambd_list)):
        ppo_train_1_expert(env, env_steps_ppo, lambd_list[i], experts_filenames[i])



def ppo_train_1_expert(env, env_steps_ppo, lambd, filename):

    # Init WandB & Parameters
    wandb.init(project='PPO', config={
        'env_id': env,
        #'env_steps': 9e6,
        'env_steps': env_steps_ppo,
        'batchsize_ppo': 12,
        'n_workers': 12,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.25,
        'lambd': lambd,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
        }, 
        reinit=True)
    config = wandb.config

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)
        scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]

        train_ready = dataset.write_tuple(states, actions, scalarized_rewards, done, log_probs, rewards, gamma=config.gamma)

        if train_ready:
            update_policy_v3(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs, config.entropy_reg, wandb)
            objective_logs = dataset.log_objectives()
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

        #vec_env.close()
        # torch.save(ppo.state_dict(), filename)
        save_data(ppo, filename)
