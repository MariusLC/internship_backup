import numpy as np
from moral.ppo import *
from moral.airl import *
from moral.active_learning import *
from moral.preference_giver import *
from envs.gym_wrapper import *
from utils.evaluate_ppo import *
from utils.save_data import *

def merge(a,b):
    result = []
    count = 0
    while len(a) > 0 and len(b) > 0:
        if a[0] <= b[0]:   
            result.append(a[0])
            a.remove(a[0])
        else:
            result.append(b[0])
            b.remove(b[0])
            count += len(a)
    if len(a) == 0:
        result = result + b
    else:
        result = result + a
    return result, count

def merge(arr, n):
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (arr[i] > arr[j]):
                inv_count += 1
  
    return inv_count

# sorted(range(len(s)), key=lambda k: s[k])

a = np.array(["a","b","c","d"])
b = np.array(["b","c","d","a"])
c = np.array(["c","d","b","a"])

inv = [np.where(a==k) for k in b]
print(inv)
print(merge(inv, len(inv)))

inv = [np.where(a==k) for k in c]
print(inv)
print(merge(inv, len(inv)))




###############
# INIT ENV
###############
env_id = "randomized_v3"
vec_env = VecEnv(env_id, 3)
states = vec_env.reset()
states_tensor = torch.tensor(states).float().to(device)

# Fetch Shapes
n_actions = vec_env.action_space.n
obs_shape = vec_env.observation_space.shape
state_shape = obs_shape[:-1]
in_channels = obs_shape[-1]

eth_norm = "v6"
non_eth_norm = "v3"

###############
# INIT AGENT
###############
print('Initializing and Normalizing Rewards...')
ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
optimizer = torch.optim.Adam(ppo.parameters(), lr=0.01)

###############
# INIT AIRL DISCRIMINATORS & GENERATORS, EVALUATE NORMALISATION POINTS
###############
non_eth_expert_filename = "generated_data/v3/[1, 0, 0, 0]/expert.pt"
discriminators_filenames = ["generated_data/v3/[0, 1, 0, 1]/discriminator.pt", "generated_data/v3/[0, 0, 1, 1]/discriminator.pt"]
generators_filenames = ["generated_data/v3/[0, 1, 0, 1]/generator.pt", "generated_data/v3/[0, 0, 1, 1]/generator.pt"]
nb_experts = len(discriminators_filenames)
discriminator_list = []
generator_list = []
rand_agent = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
non_eth_expert = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
non_eth_expert.load_state_dict(torch.load(non_eth_expert_filename, map_location=torch.device('cpu')))
for i in range(nb_experts):
    discriminator_list.append(Discriminator(state_shape=state_shape, in_channels=in_channels).to(device))
    discriminator_list[i].load_state_dict(torch.load(discriminators_filenames[i], map_location=torch.device('cpu')))
    generator_list.append(PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device))
    generator_list[i].load_state_dict(torch.load(generators_filenames[i], map_location=torch.device('cpu')))
    args = discriminator_list[i].estimate_normalisation_points(eth_norm, rand_agent, generator_list[i], env_id, 0.999, steps=1000) # tests
    discriminator_list[i].set_eval()

# PPO DATASET
dataset = TrajectoryDataset(batch_size=20, n_workers=3)

dataset.estimate_normalisation_points(non_eth_norm, non_eth_expert, env_id, steps=1000)

preference_giver = PreferenceGiverv3_no_null([1,3,1])

w_posterior_mean = [0.5, 0.5, 0.5]

train_ready = False
while not train_ready:
    # Environment interaction
    actions, log_probs = ppo.act(states_tensor)
    next_states, rewards, done, info = vec_env.step(actions)

    # Fetch AIRL rewards
    airl_state = torch.tensor(states).to(device).float()
    airl_next_state = torch.tensor(next_states).to(device).float()

    airl_rewards_list = []
    for j in range(nb_experts):
        airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, 0.999, eth_norm).squeeze(1).detach().cpu().numpy() * [0 if i else 1 for i in done])

    print("airl_rewards_list = ", airl_rewards_list)
    airl_rewards_array = np.array(airl_rewards_list)
    new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]
    print("new_airl_rewards = ", new_airl_rewards)
    print("rewards = ", rewards)
    train_ready = dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)

    # Prepare state input for next time step
    states = next_states.copy()
    states_tensor = torch.tensor(states).float().to(device)

dataset.compute_scalarized_rewards(w_posterior_mean, non_eth_norm, None)
a = preference_giver.evaluate_weights_inversions(10, w_posterior_mean, dataset.trajectories)
print(a)