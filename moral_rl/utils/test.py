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

def printing_correctly(filename, action, obs, reward, discrim_advantages, discount, done):
    f = open(filename, "a")
    infos = [{'board' : [{'player_pos' : [tuple(a) for a in np.swapaxes(np.where(obs.board == 80), 0, 1)], 
                        'citizen_pos' : [tuple(a) for a in np.swapaxes(np.where(obs.board == 67), 0, 1)],
                        'delivery_pos' : [tuple(a) for a in np.swapaxes(np.where(obs.board == 70), 0, 1)]}],
            'action' : action,
            # 'action' : action[0],
            'reward' : reward,
            'discrim_eval' : discrim_advantages.item(),
            'discount' : discount,
            'done': done}]
    f.write("\n\n"+str(infos))
    f.close()

def printing_correctly_manager(filename, action, obs, reward, discrim_advantages, discrim_rewards, log_probs, discount, done):
    f = open(filename, "a")
    infos = [{'board' : [{'player_pos' : [tuple(a) for a in np.swapaxes(np.where(obs.board == 80), 0, 1)], 
                        'citizen_pos' : [tuple(a) for a in np.swapaxes(np.where(obs.board == 67), 0, 1)],
                        'delivery_pos' : [tuple(a) for a in np.swapaxes(np.where(obs.board == 70), 0, 1)]}],
            'action' : action,
            # 'action' : action[0],
            'reward' : reward,
            'discrim_eval' : discrim_advantages.item(),
            'discrim_rewards' : discrim_rewards.item(),
            'log_probs' : log_probs[0],
            'discount' : discount,
            'done': done}]
    f.write("\n\n"+str(infos))
    f.close()


class Action_evaluator():
    def __init__(self, discrim, env, filename):
        self.discrim = discrim
        self.env = env
        self.filename = filename

    def initial_state(self, obs):
        states = self.env._obs_to_np_array(obs)
        states_tensor = torch.tensor(states).float().to(device)
        self.state=states_tensor

    def act(self, action):
        # new step fct to have infos for discrim and the ui
        obs, rewards, discount, next_state, done, info = env.step_demo(action)
        next_state_tensor = torch.tensor(next_state).to(device).float()
        discrim_advantages = self.discrim.forward(self.state, next_state_tensor, GAMMA)

        printing_correctly(self.filename, action, obs, rewards, discrim_advantages, discount, done)

        # f = open(self.filename, "a")
        # f.write("\nactions picked = "+ str(action))
        # f.write("\nobs = "+ str(obs))
        # f.write("\nrewards = "+ str(rewards))
        # f.write("\ndiscount = "+ str(discount))
        # f.write("\ndiscrim_advantages = "+ str(discrim_advantages))
        # # f.write("\ndiscrim_rewards = "+ str(discrim_rewards))
        # f.write("\ndone = "+ str(done))
        # f.write("\ninfo = "+ str(info))
        # f.close()

        self.state = next_state_tensor

        return obs, rewards

class Action_manager(Action_evaluator):
    def __init__(self, discrim, env, filename, ppo):
        super().__init__(discrim, env, filename)
        self.ppo = ppo

    def act(self):
        action, log_probs = self.ppo.act(self.state)
        obs, rewards, discount, next_state, done, info = env.step_demo(action)
        action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        next_state_tensor = torch.tensor(next_state).to(device).float()
        discrim_advantages, discrim_rewards = self.discrim.predict_reward_2(self.state, next_state_tensor, GAMMA, action_prob)

        printing_correctly_manager(self.filename, action, obs, rewards, discrim_advantages, discrim_rewards, log_probs, discount, done)

        # f = open(self.filename, "a")
        # f.write("\nactions picked = "+ str(actions))
        # f.write("\nstate = "+ str(self.state))
        # f.write("\nrewards = "+ str(rewards))
        # f.write("\ndiscount = "+ str(discount))
        # f.write("\ndiscrim_advantages = "+ str(discrim_advantages))
        # f.write("\ndiscrim_rewards = "+ str(discrim_rewards))
        # f.write("\ndone = "+ str(done))
        # f.write("\ninfo = "+ str(info))
        # f.close()

        self.state = next_state_tensor

        return obs, rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo', help='Record demonstrations',
                        action='store_true')
    args = parser.parse_args()

    c = load_config(CONFIG_PATH, CONFIG_FILENAME)

    # Create Environment
    env = make_env_demo(c["env_rad"]+c["env"], 0, (int(str(time.time()).replace('.', '')[-8:])))()

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

    # discriminator
    discrim = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    discrim.load_state_dict(torch.load(discriminator_filename, map_location=torch.device('cpu')))
    optimizer_discriminator = torch.optim.Adam(discrim.parameters(), lr=5e-5)

    # # generator
    # gene_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    # gene_policy.load_state_dict(torch.load(generator_filename, map_location=torch.device('cpu')))

    if c["demo"]:
        # experts
        expert_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
        expert_policy.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))
        policy = Action_manager(discrim, env, FILENAME, expert_policy)
        keys_to_actions={'e': 9, 'E': 9, 'a': None}
    else:
        policy = Action_evaluator(discrim, env, FILENAME)
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                     curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                     'z': 5,
                     's': 6,
                     'q': 7,
                     'd': 8,
                     -1: 4,
                     'e': 9, 'E': 9,}

    # define wether the game has a time limit between 2 actions
    if c["delayed"] :
        delay = 1000
    else :
        delay = None

    # we have to give a keys to actions to all possible actions from env ?
    ui = CursesUi_Marius(
        policy=policy,
        keys_to_actions=keys_to_actions,
        delay=delay,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(env.game)