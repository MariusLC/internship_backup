# internship_backup

## Auteurs
Auteurs: Marius Le Chapelier  
Encadrants: Aurélie Beynier, Nicolas Maudet, Paolo Viappiani (et partiellement Ann Nowé)

## Description
Stage de fin d'étude sur le développement d'un modèle d'apprentissage par renforcement multi-objectifs pour répondre à des problèmes éthiques.

Date: Avril - Septembre 2022

## Dependencies
* wandb
* tqdm
* pytorch \>= 1.7.0
* numpy \>= 1.20.0
* scipy \>= 1.1.0
* pycolab == 1.2

## Weights and Biases
Our code depends on ![Weights and Biases](https://wandb.ai/) for visualizing and logging results during training.
As a result, we call `wandb.init()`, which will prompt to add an API key for linking the training runs with your 
personal wandb account. This can be done by pasting the `WANDB_API_KEY` into the respective box when running 
the code for the first time.

## Environments
Our gridworld Delivery: `randomized_v3.py`, build on the ![Pycolab](https://github.com/deepmind/pycolab) game engine with a custom wrapper
to provide similar functionality as the `gym` ![environments](https://github.com/openai/gym). This engine 
comes with a user interface and any `environment` can be played in the console using `python environment.py` 
with arrow keys and `w`, `a`, `s`, `d` as controls.

## Training
There are four training scripts for

* manually training a PPO agent on custom rewards (`ppo_train.py`),
* training AIRL on a single expert dataset (`airl_train.py`),
* active MORL with custom/automatic preferences (`moral_train.py`) and
* training DRLHP with custom/automatic preferences (`drlhp_train.py`).

Each script have a corresponding configuration file (on the `config` folder) where you can set every hyperparameter.

the architecture of the code is simple : for each of the 4 files above, there is one corresponding file for the training loop and one for all the base code of the algorithms. For example, `ppo_train_not_main.py` corresponds to the `ppo_train.py` training loop file, and `ppo.py` corresponds to the base code.
