from envs.randomized_v1_test import *
from utils.load_config import *
from envs.gym_wrapper import *
from moral.airl import *
import torch



# from envs.gym_wrapper import *
# from moral.airl_train_not_main import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_Ui.yaml"

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo', help='Record demonstrations',
                        action='store_true')
    args = parser.parse_args()

    c = load_config(CONFIG_PATH, CONFIG_FILENAME)

    # Create Environment
    vec_env = make_env(c["env_rad"]+c["env"], 0, (int(str(time.time()).replace('.', '')[-8:])))()
    # vec_env = VecEnv(c["env_rad"]+c["env"], 12)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    vanilla_path = ""
    if c["vanilla"]:
        vanilla_path = c["vanilla_path"]

    path = c["data_path"]+c["env_path"]+vanilla_path+str(c["expert_weights"])+"/"

    expert_filename = path+c["expe_path"]+c["model_ext"]
    generator_filename = path+c["gene_path"]+c["model_ext"]
    discriminator_filename = path+c["disc_path"]+c["model_ext"]
    discriminator_old_filename = path+c["disc_path"]+"2"+c["model_ext"]

    demos_filename = path+c["demo_path"]+c["demo_ext"]
    demos_filename = path+c["demo_path"]+"2"+c["demo_ext"]
    rand_filename = path+c["demo_path"]+c["rand_path"]+c["demo_ext"]
    gene_demos_filename = path+c["demo_path"]+c["gene"]+c["demo_ext"]
    
    # print(demos_filename)

    # # experts
    # expert_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    # expert_policy.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))

    # discriminator
    discrim = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    discrim.load_state_dict(torch.load(discriminator_filename, map_location=torch.device('cpu')))
    optimizer_discriminator = torch.optim.Adam(discrim.parameters(), lr=5e-5)

    # # old discriminator
    # discrim_old_list = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    # discrim_old_list.load_state_dict(torch.load(discriminator_old_filename, map_location=torch.device('cpu')))
    # optimizer_old_discriminator = torch.optim.Adam(discrim_old_list.parameters(), lr=5e-5)

    # # generator
    # gene_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    # gene_policy.load_state_dict(torch.load(generator_filename, map_location=torch.device('cpu')))


    if (c["env_rad"]+c["env"]) == "randomized_v1_test":
    	# this is calling the game
    	# randomized_v1_test.main(discrim)
    	main(False, discrim)