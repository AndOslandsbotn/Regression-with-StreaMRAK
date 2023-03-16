from solvers.trainer import FalkonTrainer
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
    filename = 'test_falkon'

    training_data, validation_data = generate_toy_data(seed=1)

    # Select landmarks
    landmarks = np.array([0.1, 0.5]).reshape(-1, 1)

    falkon_trainer = FalkonTrainer(config_solver, config_training)
    falkon_trainer.add_training_data(training_data)
    falkon_trainer.add_validation_data(validation_data)
    falkon_trainer.select_random_landmarks()
    falkon_trainer.append_specific_landmarks(landmarks)
    falkon_trainer.train(filename)



