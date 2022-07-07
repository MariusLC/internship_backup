from envs.gym_wrapper import *
from stable_baselines3.common.vec_env import SubprocVecEnv
from moral.ppo import *
from tqdm import tqdm
import torch
from drlhp.preference_model import *
from moral.preference_giver import *
import wandb
import pickle
import argparse
from utils.save_data import *


if __name__ == '__main__':

    after_moral_filename = "generated_data/v3/after_moral/test.pt"

    # Pretrained MORAL agent that we want to teach action differences
    # moral_agent_filename = "generated_data/v3/moral_agents/[[0, 1, 0, 1], [0, 0, 1, 1]]131_new_norm_v6_v3.pt"
    moral_agent_filename = "generated_data/v3/moral_agents/[[0, 1, 0, 1], [0, 0, 1, 1]]131_norm_v6_v4_div2.pt"

    # Pretrained preference model estimating expert preferences
    # preference_model_filename = "generated_data/v3/pref_model/1000q_ParetoDom.pt"
    preference_model_filename = "generated_data/v3/pref_model/5000q_50b_200e_1>0>2>3.pt"

    # Config
    wandb.init(project='AFTER_MORAL', config={
        'env_id': 'randomized_v3',
        # 'ratio': ratio,
        'env_steps': 8e6,
        'batchsize_ppo': 12,
        # 'batchsize_preference': 12,
        # 'n_queries': 1000,
        # 'update_reward_freq': 50,
        # 'preference_warmup': 1,
        # 'pretrain': 1000,
        'n_workers': 12,
        'lr_ppo': 3e-4,
        # 'lr_reward': 3e-5,
        'entropy_reg': 0.05,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5,
        "env_dim": 4,
        'after_moral_filename' : after_moral_filename,
        'moral_agent_filename' : moral_agent_filename,
        'preference_model_filename' : preference_model_filename,
    })
    config = wandb.config
    env_steps = int(config.env_steps / config.n_workers)
    # query_freq = int(env_steps / (config.n_queries + 2))

    # Create Environment
    vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(config.n_workers)])
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    ppo.load_state_dict(torch.load(moral_agent_filename, map_location=torch.device('cpu')))

    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    # On peut faire un preference model avec en entrée les retours d'actions ou les etats de l'environnement, a voir ...
    preference_model = PreferenceModelTEST(config.env_dim).to(device)
    preference_model.load_state_dict(torch.load(preference_model_filename, map_location=torch.device('cpu')))

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)
        preference_state = torch.tensor(states).to(device).float()
        preference_rewards = preference_model.evaluate_action(rewards).squeeze(1)
        preference_rewards = list(preference_rewards.detach().cpu().numpy())

        train_ready = dataset.write_tuple(states, actions, preference_rewards, done, log_probs, rewards)

        if train_ready:

            # objective_logs = dataset.log_objectives()
            # for i in range(objective_logs.shape[1]):
            #     wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            # for ret in dataset.log_rewards():
            #     wandb.log({'Returns': ret})

            # Log Objectives
            obj_ret = dataset.log_objectives()
            obj_ret_logs = np.mean(obj_ret, axis=0)
            for i, ret in enumerate(obj_ret_logs):
                wandb.log({'Obj_' + str(i): ret}, step=t*config.n_workers)

            # Log total weighted sum
            wandb.log({'Returns mean': np.mean(dataset.log_rewards())}, step=t*config.n_workers)


            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)

            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
        # save_data(ppo, after_moral_filename)

    save_data(ppo, after_moral_filename)
