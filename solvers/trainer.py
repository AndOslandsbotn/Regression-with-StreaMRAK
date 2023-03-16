from solvers.falkon import Falkon
from solvers.streamrak import Streamrak
from solvers.cover_tree import estimate_span
from utilities.util import timer_func, ndarray_to_list

from sklearn.metrics import mean_squared_error
import numpy as np
from pathlib import Path
import os
import json

from definitions import ROOT_DIR

class TrainerBase():
    def __init__(self):
        self.training_data = None
        self.validation_data = None

    def add_training_data(self, training_data):
        self.training_data = training_data

    def add_validation_data(self, validation_data):
        self.validation_data = validation_data

    def save_model(self, model, filename):
        """Saves model to file"""
        results_folder = os.path.join(ROOT_DIR, 'StoredModels')
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        model = ndarray_to_list(model)

        training_info = {}
        training_info['model'] = model
        training_info['time'] = self.time_usage

        model_filename = filename+'.json'
        with open(os.path.join(results_folder, model_filename), 'w') as f:
            f.write(json.dumps(training_info, indent=4))
        return

class FalkonTrainer(TrainerBase):
    def __init__(self, config_solver, config_training):
        super().__init__()
        self.config = config_training
        self.falkon = Falkon(config_solver)
        self.time_usage = {}

    @timer_func
    def find_optimal_bw(self):
        bwmax, bwmin, size = self.config['falkon']['bw_grid']
        bw_grid = np.logspace(bwmax, bwmin, size)

        mse = []
        for i, bw in enumerate(bw_grid):
            print(f"\n Find optimal bw iter {i}/{len(bw_grid)}")
            self.falkon.set_bw(bw)
            self.falkon.train(self.training_data['x'], self.training_data['y'])
            yval_pred = self.falkon.predict(self.validation_data['x'])
            mse.append(mean_squared_error(self.validation_data['y'], yval_pred))
        idx = np.argmin(mse)
        return bw_grid[idx]

    def select_random_landmarks(self):
        self.falkon.select_random_landmarks(self.training_data['x'])

    def append_specific_landmarks(self, landmarks):
        self.falkon.append_specific_landmarks(landmarks)

    def train(self, filename):
        # Find optimal bandwidth
        opt_bw, find_bw_time = self.find_optimal_bw()
        self.time_usage['find_bw_time'] = find_bw_time

        # Train falkon
        self.falkon.set_bw(opt_bw)
        print("Optimal bandwidth: ", opt_bw)
        _, train_time = self.falkon.train(self.training_data['x'], self.training_data['y'], timeit=True)
        model = self.falkon.get_model()
        self.time_usage['train_time'] = train_time

        self.save_model(model, filename)
        self.falkon.clear()
        return


class StreamrakTrainer(TrainerBase):
    def __init__(self, config_solver, config_training):
        self.trainer_type = 'streamrak'
        super().__init__()

        self.config = config_training
        self.streamrak = Streamrak(config_solver)
        self.time_usage = {}

    def train(self, filename):

        # Initializa cover-tree
        init_radius = estimate_span(self.training_data['x'])
        self.streamrak.initialize(init_radius)

        # Train streamrak
        _, train_time = self.streamrak.train(self.training_data['x'], self.training_data['y'], timeit=True)
        model = self.streamrak.get_model()
        self.time_usage['train_time'] = train_time

        self.save_model(model, filename)
        self.streamrak.clear()
        return
