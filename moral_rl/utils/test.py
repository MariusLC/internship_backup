from envs.randomized_v1_test import *
from utils.load_config import *
from envs.gym_wrapper import *
from moral.airl import *
import torch

from moral.ppo import *



# from envs.gym_wrapper import *
# from moral.airl_train_not_main import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_Ui.yaml"
FILENAME = "generated_data/logs_test.txt"

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

GAMMA = 0.999

class Policy():
    def __init__(self, ppo, discrim, env, filename):
        self.ppo = ppo
        self.discrim = discrim
        self.env = env
        self.filename = filename
        # self.state = state
        # self.next_state = None

    def initial_state(self, obs):
        states = self.env._obs_to_np_array(obs)
        states_tensor = torch.tensor(states).float().to(device)
        self.state=states_tensor

    def act(self):
        # airl_state = torch.tensor(states).to(device).float()
        # airl_next_state = torch.tensor(next_states).to(device).float()

        actions, log_probs = self.ppo.act(self.state)
        # new step fct to have infos for discrim and the ui
        obs, rewards, discount, next_state, done, info = env.step_demo(actions)
        action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        next_state_tensor = torch.tensor(next_state).to(device).float()
        discrim_advantages, discrim_rewards = self.discrim.predict_reward_2(self.state, next_state_tensor, GAMMA, action_prob)

        f = open(self.filename, "a")
        f.write("\nactions picked = "+ str(actions))
        f.write("\nstate = "+ str(self.state))
        f.write("\nrewards = "+ str(rewards))
        f.write("\ndiscount = "+ str(discount))
        f.write("\ndiscrim_advantages = "+ str(discrim_advantages))
        f.write("\ndiscrim_rewards = "+ str(discrim_rewards))
        f.write("\ndone = "+ str(done))
        f.write("\ninfo = "+ str(info))
        f.close()

        self.state = next_state_tensor

        # # Prepare state input for next time step
        # states = next_states.copy()
        # states_tensor = torch.tensor(states).float().to(device)

        return obs, rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo', help='Record demonstrations',
                        action='store_true')
    args = parser.parse_args()

    c = load_config(CONFIG_PATH, CONFIG_FILENAME)

    # Create Environment
    env = make_env_demo(c["env_rad"]+c["env"], 0, (int(str(time.time()).replace('.', '')[-8:])))()
    # vec_env = VecEnv(c["env_rad"]+c["env"], 12)
    # states = env.reset()
    # states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    vanilla_path = ""
    if c["vanilla"]:
        vanilla_path = c["vanilla_path"]

    path = c["data_path"]+c["env_path"]+vanilla_path+str(c["expert_weights"])+"/"

    expert_filename = path+c["expe_path"]+c["model_ext"]
    generator_filename = path+c["gene_path"]+c["model_ext"]
    discriminator_filename = path+c["disc_path"]+c["model_ext"]

    demos_filename = path+c["demo_path"]+c["demo_ext"]
    rand_filename = path+c["demo_path"]+c["rand_path"]+c["demo_ext"]
    gene_demos_filename = path+c["demo_path"]+c["gene"]+c["demo_ext"]

    # experts
    expert_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    expert_policy.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))

    # discriminator
    discrim = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    discrim.load_state_dict(torch.load(discriminator_filename, map_location=torch.device('cpu')))
    optimizer_discriminator = torch.optim.Adam(discrim.parameters(), lr=5e-5)

    # # generator
    # gene_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    # gene_policy.load_state_dict(torch.load(generator_filename, map_location=torch.device('cpu')))

    policy = Policy(expert_policy, discrim, env, FILENAME)

    if not c["demo"]:

        # this is calling the game
        # randomized_v1_test.main(discrim)
        main(False, c["delayed"], policy)

    else :

        # define wether the game has a time limit between 2 actions
        if c["delayed"] :
            delay = 1000
        else :
            delay = None

        # we have to give a keys to actions to all possible actions from env ?
        ui = CursesUi_Marius(policy=policy,
            keys_to_actions={'a':0,
                             -1: 4,
                             'e': 9, 'E': 9},
            delay=delay,
            colour_fg=WAREHOUSE_FG_COLOURS)

        # Let the game begin!
        ui.play(env.game)