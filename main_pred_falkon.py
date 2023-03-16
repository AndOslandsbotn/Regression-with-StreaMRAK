from solvers.predicter import FalkonPredicter
from sklearn.metrics import mean_squared_error
from utilities.util import generate_toy_data

from definitions import CONFIG_SOLVERS_PATH
from config.yaml_functions import yaml_loader
config_solver = yaml_loader(CONFIG_SOLVERS_PATH)

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from definitions import CONFIG_MATPLOTSTYLE_PATH
plt.style.use(CONFIG_MATPLOTSTYLE_PATH)

if __name__ == '__main__':
    filename_model = 'test_falkon'
    filename_store_pred = 'pred_test_falkon'

    test_data, _ = generate_toy_data(seed=3)

    falkon_predicter = FalkonPredicter(config_solver)
    falkon_predicter.add_existing_model(model_name=filename_model)
    falkon_predicter.predict(test_data)
    falkon_predicter.save(filename_store_pred)

    yts_pred = falkon_predicter.get_pred()

    sort_idx = np.argsort(test_data['x'], axis=0)[:, 0]
    test_data['x'] = test_data['x'][sort_idx]
    test_data['y'] = test_data['y'][sort_idx]
    yts_pred = yts_pred[sort_idx]

    mse = mean_squared_error(test_data['y'], yts_pred)
    print("Mean square error: ", mse)

    plt.figure()
    plt.plot(test_data['x'], test_data['y'], label = 'ref')
    plt.plot(test_data['x'], yts_pred, linewidth=1.8, linestyle='--', label='pred')
    plt.legend()
    plt.show()
