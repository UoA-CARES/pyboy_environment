# pyboy_environment
Envrionment for pyboy games for Reinforcement Learning training

## Installation Instructions

`git clone` the repository into your desired directory on your local machine

Run `pip3 install -r requirements.txt` in the **root directory** of the package

To make the module **globally accessible** in your working environment run `pip3 install --editable .` in the **project root**

# Usage
This package provides the baseline code for the pyboy environments - you run these envrionments through gymnasium_envrionment.

`train.py` takes in hyperparameters that allow you to customise the training run enviromment – OpenAI or DMCS Environment - or RL algorithm. Use `python3 train.py -h` for help on what parameters are available for customisation.

An example is found below for running on the pyboy environments with TD3 through console
```
python3 train.py run --gym pyboy --task mario TD3
```

# Games
Environment running Gameboy games utilising the pyboy wrapper: https://github.com/UoA-CARES/pyboy_environment 

These games are run through the generalised gymnaisum environment: https://github.com/UoA-CARES/gymnasium_envrionments

## Mario

```
python3 train.py run --gym pyboy --domain mario --task run SACAE
```

<p align="center">
    <img src="./media/mario.png" style="width: 40%;" />
</p>

## Pokemon

```
python3 train.py run --gym pyboy --domain pokemon --task catch SACAE
```

<p align="center">
    <img src="./media/pokemon.png" style="width: 40%;"/>
</p>