from solvers.falkon import FalkonSolver
from solvers.cover_tree import BasicCoverTree
from utilities.choose_kernel import choose_kernel
from utilities.util import timer_func

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

class Lpkrr():
    """Laplacian pyramid (LP) formulation of kernel ridge regression (KRR)
    """
    def __init__(self, config):
        self.config = config
        self.model = {}
        self.lvlcursor = 0
        self.pred_dict = {}
        self.res_dict = {}
        self.kernel_obj = choose_kernel(config['falkon']['kernelType'])
        self.solver = FalkonSolver(config)
        self.max_ram_usage = config['streamrak']['max_ram_usage'] # max ram usage in Bytes

    def __str__(self):
        print(f"Kernel type {self.config['falkon']['kernelType']}")
        print(f"number of levels trained {self.lvlcursor}")
        for lvl in range(0, self.lvlcursor):
            lm, bw, coef = self.model[f'lvl{lvl}']
            print(f'Num lm: {len(lm)}\n Bandwidth: {bw}')
        return

    def predict(self, x, timeit=False):
        """Returns prediction time if timeit == True"""
        if timeit:
            return self.predict_with_timeit(x)
        else:
            y_pred, pred_time = self.predict_with_timeit(x)
            return y_pred

    def train(self, x, y, timeit=False):
        """Returns training time if timeit == True"""
        if timeit:
            return self.train_with_timeit(x, y)
        else:
            self.train_with_timeit(x, y)
            return None

    @timer_func
    def train_with_timeit(self, x, y, lmtr_list, bw_list):
        for lvl in range(0, self.config['nlvls']):
            print(f'Training lvl {self.lvlcursor}')

            lm = lmtr_list[lvl]
            bw = bw_list[lvl]
            if lvl == 0:
                y_res = y
            else:
                y_res = self.calc_residual(x, y)
            self.res_dict[f'lvl{lvl}'] = y_res

            n = len(x)
            Kmm = self.solver.calcKmm(lm, bw)
            #KnmTKnm = self.solver.calcKnmTKnm(x, lm, bw)
            #Zm = self.solver.calcZm(x, y_res, lm, bw)
            KnmTKnm, Zm = self.solve_in_batches(x, y_res, lm, bw)
            Zm = (1 / n) * Zm
            coef = self.solver.fit(Kmm, KnmTKnm, Zm, n)
            self.model[f'lvl{lvl}'] = (lm, bw, coef)
            self.lvlcursor = lvl+1
        return

    def solve_in_batches(self, x, y, lm, bw):
        n, ydim = y.shape
        m, xdim = lm.shape
        x_batches = self.split_in_batches(x, n, m, xdim)
        y_batches = self.split_in_batches(y, n, m, xdim)

        KnmTKnm = np.zeros((m, m))
        Zm = np.zeros((m, ydim))
        for batch_nr, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            print(f"Batch nr {batch_nr+1}/{len(x_batches)}")
            KnmTKnm = KnmTKnm + self.solver.calcKnmTKnm(x_batch, lm, bw)
            Zm = Zm + self.solver.calcZm(x_batch, y_batch, lm, bw)
        return KnmTKnm, Zm

    def split_in_batches(self, x, n, m, dim):
        """
        Knm requires n x d x m x 8 (float64) bytes. We find max n such that we dont used
        more ram than: max_ram_usage
        """
        n_max = self.max_ram_usage/(m*dim*8)
        num_batches = np.ceil(n/n_max)
        x = np.array(np.array_split(x, num_batches))
        return x

    @timer_func
    def predict_with_timeit(self, x, maxlvl=None):
        """ Make a prediction y = f(x) using the currently available model
        Parameters
        ----------
        :param x: Data points, ndarray with shape = (n_training_samples, ambient_dim)
        :param maxlvl: Level up to which we want to make predictions
        Returns
        -------
        :return: Prediction y
        """

        if self.lvlcursor == 0:
            print("No fitted levels")
            return None

        if maxlvl != None and maxlvl <= self.lvlcursor:
            predlvl = maxlvl
        else:
            predlvl = self.lvlcursor

        for lvl in range(0, predlvl):
            if lvl == 0:
                y_pred = self.predict_in_batches(x, lvl)
                self.pred_dict[f'lvl{lvl}'] = y_pred
            else:
                y_pred = y_pred + self.predict_in_batches(x, lvl)
                self.pred_dict[f'lvl{lvl}'] = y_pred
        return y_pred

    def predict_in_batches(self, x, lvl):
        lm, bw, coef = self.model[f'lvl{lvl}']
        n, xdim = x.shape
        m, xdim = lm.shape
        x_batches = self.split_in_batches(x, n, m, xdim)

        for batch_nr, x_batch in enumerate(x_batches):
            print(f"Batch nr {batch_nr + 1}/{len(x_batches)}")
            if batch_nr == 0:
                y_pred = self.solver.predict(x_batch, lm, coef, bw)
            else:
                y_pred = np.concatenate((y_pred, self.solver.predict(x_batch, lm, coef, bw)), axis=0)
        return y_pred

    def calc_residual(self, x, y):
        """
        Calculates residual between y-target and prediction made by currently available model

        Parameters
        ----------
        :param x: Data points, ndarray with shape = (n_training_samples, ambient_dim])
        :param y: Target value, ndarray with shape = (n_training_samples, n_target_functions)

        Returns
        -------
        :return: Residual after prediction
        """
        return y - self.predict(x)

    def get_pred(self, lvl=None):
        """Returns prediction at lvl"""
        if lvl is None:
            return self.pred_dict
        else:
            assert lvl < self.coverTree.tree_depth, f"Lvl must be smaller than {self.coverTree.tree_depth}"
            return self.pred_dict[f'lvl{lvl}']

    def get_res(self, lvl=None):
        """Returns residual at lvl"""
        if lvl is None:
            return self.res_dict
        else:
            assert lvl < self.coverTree.tree_depth, f"Lvl must be smaller than {self.coverTree.tree_depth}"
            return self.res_dict[f'lvl{lvl}']

    def get_model(self):
        return self.model

    def add_trained_model(self, model):
        self.model = model
        self.lvlcursor = len(list(self.model.keys()))
        self.coverTree = BasicCoverTree(self.config, init_radius=self.model['lvl0'][1])

    def clear(self):
        """Clears the model"""
        self.model = {}
        self.lvlcursor = 0
        self.pred_dict = {}
        self.res_dict = {}


class Streamrak(Lpkrr):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lmfactor = float(config['streamrak']['lmfactor'])
        self.debug = config['general']['debug']

        assert isinstance(self.lmfactor, float), f'lmfactor must be float'
        assert isinstance(self.debug, bool), f'debug must be bool'

        self.coverTree = None

    def initialize(self, init_radius):
        self.coverTree = BasicCoverTree(self.config, init_radius)

    def build_cover_tree(self, X, Y):
        for x, y in zip(X, Y):
            self.coverTree.insert(x, y)

    def predict(self, x, timeit=False):
        """Returns prediction time if timeit == True"""
        if timeit:
            return self.predict_with_timeit(x)
        else:
            y_pred, pred_time = self.predict_with_timeit(x)
            return y_pred

    def train(self, x, y, timeit=False):
        """Returns training time if timeit == True"""
        if timeit:
            return self.train_with_timeit(x, y)
        else:
            self.train_with_timeit(x, y)
            return None

    @timer_func
    def train_with_timeit(self, x, y):
        self.build_cover_tree(x, y)
        if self.debug:
            self.coverTree.__str__()

        print("Tree depth: ", self.coverTree.tree_depth)
        for lvl in range(0, self.coverTree.tree_depth):
            print(f'\nTraining lvl {self.lvlcursor}')

            lm = self.select_landmarks(lvl)
            print(f"Number of landmarks {len(lm)}")
            bw = self.coverTree.get_radius(lvl+1)

            # Added this to test
            if lvl > 6: # We re-use bw at lvl 6 for all deeper levels
                bw = self.model[f'lvl{6}'][1]

            if lvl == 0:
                y_res = y
            else:
                y_res = self.calc_residual(x, y)
            self.res_dict[f'lvl{lvl}'] = y_res

            n = len(x)
            Kmm = self.solver.calcKmm(lm, bw)
            KnmTKnm, Zm = self.solve_in_batches(x, y_res, lm, bw)
            Zm = (1 / n) * Zm
            coef = self.solver.fit(Kmm, KnmTKnm, Zm, n)
            self.model[f'lvl{lvl}'] = (lm, bw, coef)
            self.lvlcursor = lvl + 1
        return

    def select_landmarks(self, lvl):
        lm = np.array(self.coverTree.get_centers(lvl + 1))
        c = int(self.lmfactor * np.sqrt(len(lm)))
        return lm

    def clear(self):
        """Clears the model"""
        self.coverTree.clear()
        self.coverTree = None
        super().clear()




