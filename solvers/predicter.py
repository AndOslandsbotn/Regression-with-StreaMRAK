from solvers.falkon import Falkon
from solvers.streamrak import Streamrak
from utilities.util import load_json_data, list_to_ndarray

from definitions import ROOT_DIR
import os
from pathlib import Path
import numpy as np

class BasePredicter():
    def __init__(self):
        self.model = None
        self.xts = None
        self.yts = None
        self.prediction_time = None
        self.model_name = None

    def get_pred(self):
        return self.yts_pred

    def add_existing_model(self, model_name):
        """Loads existing model
        :param model_name: Name of model
        """
        self.model_name = model_name
        model_folder = os.path.join(ROOT_DIR, 'StoredModels')
        model_filename = model_name+'.json'
        trained_model = load_json_data(os.path.join(model_folder, model_filename))
        trained_model = list_to_ndarray(trained_model['model'])
        self.model.add_trained_model(trained_model)

    def predict(self, test_data):
        self.xts = test_data['x']
        self.yts = test_data['y']
        self.yts_pred, self.prediction_time = self.model.predict(self.xts, timeit=True)
        self.model.clear()

    def save(self, filename):
        results_folder = os.path.join(ROOT_DIR, 'Results', self.model_name)
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        np.savez(os.path.join(results_folder, filename),
                 xts=self.xts, yts=self.yts,
                 yts_pred=self.yts_pred, predtime = self.prediction_time)
        return

class FalkonPredicter(BasePredicter):
    def __init__(self, config_solver):
        self.model = Falkon(config_solver)

class StreamrakPredicter(BasePredicter):
    def __init__(self, config_solver):
        self.model = Streamrak(config_solver)