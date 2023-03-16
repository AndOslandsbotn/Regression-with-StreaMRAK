from solvers.trainer import StreamrakTrainer
from utilities.util import generate_toy_data

from definitions import CONFIG_SOLVERS_PATH, CONFIG_TRAINER_PATH
from config.yaml_functions import yaml_loader
config_solver = yaml_loader(CONFIG_SOLVERS_PATH)
config_training = yaml_loader(CONFIG_TRAINER_PATH)

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from definitions import CONFIG_MATPLOTSTYLE_PATH
plt.style.use(CONFIG_MATPLOTSTYLE_PATH)


if __name__ == '__main__':
    filename = 'test_streamrak'

    training_data, validation_data = generate_toy_data(seed=1)

    # Select landmarks
    streamrak_trainer = StreamrakTrainer(config_solver, config_training)
    streamrak_trainer.add_training_data(training_data)
    streamrak_trainer.add_validation_data(validation_data)
    streamrak_trainer.train(filename)



