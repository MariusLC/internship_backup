from envs.gym_wrapper import *
from utils.load_config import *
from moral.ppo_train_not_main import *
from utils.generate_demos_not_main import *
from moral.airl_train_not_main import *
from moral.moral_train_not_main import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_COMPLETE.yaml"

if __name__ == '__main__':

    c = load_config(CONFIG_PATH, CONFIG_FILENAME)

    vanilla_path = ""
    if c["vanilla"]:
        vanilla_path = c["vanilla_path"]

    query_freq = c["query_freq"]
    if c["real_params"]:
        query_freq = c["nb_steps"]/(c["n_queries"]+2)

    experts_filenames = []
    demos_filenames = []
    generators_filenames = []
    discriminators_filenames = []
    moral_filename = c["data_path"]+c["env_path"]+vanilla_path+c["moral_path"]+str(c["experts_weights"])+c["model_ext"]
    for i in range(c["nb_experts"]):
        path = c["data_path"]+c["env_path"]+vanilla_path+str(c["experts_weights"][i])+"/"
        experts_filenames.append(path+c["expe_path"]+c["model_ext"])
        demos_filenames.append(path+c["demo_path"]+c["demo_ext"])
        generators_filenames.append(path+c["gene_path"]+c["model_ext"])
        discriminators_filenames.append(path+c["disc_path"]+c["model_ext"])


    # TRAINING PPO AGENTS
    ppo_train_n_experts(c["env_rad"]+c["env"], c["env_steps_ppo"], c["experts_weights"], experts_filenames)

    # GENERATING DEMONSTRATIONS FROM EXPERTS
    generate_demos_n_experts(c["env_rad"]+c["env"], c["nb_demos"], experts_filenames, demos_filenames)

    # ESTIMATING EXPERTS REWARD FUNCTIONS THROUGH AIRL BASED ON THEIR DEMONSTRATIONS
    airl_train_n_experts(c["env_rad"]+c["env"], c["env_steps_airl"], demos_filenames, generators_filenames, discriminators_filenames)

    # ESTIMATING MORL EXPERT'S WEIGTHS THROUGH MORAL
    
    # On the original code, utopia point in moral phase was calculated wrt the experts policies. 
    # But it seems more logical if it was wtr generators from the airl process. 
    # So we can choose by changing the parameter geneORexpert in the config file.
    gene_or_expe_filenames = generators_filenames
    if c["geneORexpert"]:
        gene_or_expe_filenames = experts_filenames

    moral_train_n_experts(c["env_rad"]+c["env"], c["ratio"], c["env_steps_moral"], query_freq, gene_or_expe_filenames, discriminators_filenames, moral_filename)