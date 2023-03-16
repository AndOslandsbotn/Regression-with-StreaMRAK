import scipy.linalg as lalg
import numpy as np

from utilities.util import timer_func
from utilities.choose_kernel import choose_kernel
from tqdm import tqdm

class Falkon():
    """ This class generates the Falkon trainer and prediction interface that allows training of a model and prediction
    with a trained model

    ## Variables
        - config: solver configurations
        - bw: bandwidth of kernel model
        - lms: the landmarks (Nystrom sub-samples)
        - nlm: number of landmarks
        - model:
            - regression model (bw, lm, coef)

        - kernel_obj: kernel used in regression model
        - lmfactor: number of landmarks is selected as nlm = lmfactor*sqrt(num training samples)
        - solver: instance of FalkonSolver class
        - max_ram_usage: max ram usage in Bytes

    ## Public methods
        # train
            - trains model on training data
        # predict
            - predicts with recieved data
        # add_trained_model
            - Adds a trained falkon model to the model variable
        # set_bw
            - sets the bandwidth of the kernel
        # append_specific_landmarks
            - if specific landmarks should be added
        # select_random_landmarks
            - selects landmarks uniformly from training data
        # clear
            -clears model

    ## Private methods
        # split_in_batches
        # solve_in_batches
        # train_with_timeit
        # predict_with_timeit

    ## Get functions
        # get_model
    """
    def __init__(self, config):
        self.config = config
        self.model = {}
        self.bw = None
        self.nlm = None
        self.lms = None

        self.lvlcursor = 0
        self.kernel_obj = choose_kernel(config['falkon']['kernelType'])
        self.lmfactor = config['falkon']['num_landmarks']
        self.solver = FalkonSolver(config)
        self.max_ram_usage = config['falkon']['max_ram_usage']  # max ram usage in Bytes

    def clear(self):
        """Clears the model"""
        self.model = None
        self.bw = None
        self.nlm = None
        self.lms = None

    def select_random_landmarks(self, x):
        """ Selects landmarks uniformly from the data x
        :param x: n x d numpy array of data samples. n is the number of samples and d the dimension
        """
        self.nlm = int(self.lmfactor * np.sqrt(len(x)))
        lm_idx = np.random.choice(np.arange(0, len(x)), self.nlm)
        self.lms = x[lm_idx, :]

    def append_specific_landmarks(self, landmarks):
        """ Adds the landmarks to the existing landmarks
         :param landmarks: n x d numpy array of landmarks. m is the number of landmarks and d the dimension
         """
        self.nlm = len(landmarks)
        self.lms = np.concatenate((self.lms, landmarks), axis=0)

    def split_in_batches(self, x, n, m, dim):
        """
        Knm requires n x d x m x 8 (float64) bytes. We find max n such that we dont used
        more ram than: max_ram_usage
        """
        n_max = self.max_ram_usage/(m*dim*8)
        num_batches = np.ceil(n/n_max)
        x = np.array(np.array_split(x, num_batches))
        return x

    def solve_in_batches(self, x, y):
        """Trains the falkon model in batches
        :param x: n x d numpy array of training samples and
        :param y: numpy array of length n, of y=f(x).
        """

        n, ydim = y.shape
        m, xdim = self.lms.shape
        x_batches = self.split_in_batches(x, n, m, xdim)
        y_batches = self.split_in_batches(y, n, m, xdim)

        KnmTKnm = np.zeros((m, m))
        Zm = np.zeros((m, ydim))
        for batch_nr, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            print(f"Batch nr {batch_nr+1}/{len(x_batches)}")
            KnmTKnm = KnmTKnm + self.solver.calcKnmTKnm(x_batch, self.lms, self.bw)
            Zm = Zm + self.solver.calcZm(x_batch, y_batch, self.lms, self.bw)
        return KnmTKnm, Zm

    def set_bw(self, bw):
        self.bw = bw

    def get_model(self):
        return self.model

    def add_trained_model(self, model):
        self.model = model

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
            return

    @timer_func
    def train_with_timeit(self, x, y):
        Kmm = self.solver.calcKmm(self.lms, self.bw)
        KnmTKnm, Zm = self.solve_in_batches(x, y)
        Zm = (1 / self.nlm) * Zm
        coef = self.solver.fit(Kmm, KnmTKnm, Zm, self.nlm)
        self.model['lvl0'] = (self.lms, self.bw, coef)
        return

    @timer_func
    def predict_with_timeit(self, x):
        lm, bw, coef = self.model['lvl0']
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

class FalkonMatrixSystem():
    """This class implements the matrix system for the FALKON
    algorithm."""
    def __init__(self, config):
        self.kernelType = config['falkon']['kernelType']
        self.kernel_obj = choose_kernel(self.kernelType)

    def calcW(self, cholT, cholTt, cholA, cholAt, KnmTKnm, n, v, reg_param):
        """
        With the preconditioner B = 1/sqrt(n)*invers(T)*invers(A) we have
        W = transpose(B)*H*B.
        :param cholT(v): Solves Tx=v for x
        :param cholTt(v): Solves T'x=v for x
        :param cholA(v): Solves Ax=v for x
        :param cholAt(v): Solves A'x=v for x
        :param KnmTKnm: m x m matrix
        :param n: number of training points
        :param v: parameter to apply the matrix W on
        :return:
        """
        v = cholA(v)
        KnmTKnmBv = np.dot(KnmTKnm, cholT(v))
        W = (1 / n) * cholAt(cholTt(KnmTKnmBv) + n*reg_param * v)
        return W

    def calcKmm(self, landmarks, scale):
        return self.kernel_obj.calcKernel(landmarks, landmarks, scale)

    def calcKnm(self, x_tr, landmarks, scale):
        return self.kernel_obj.calcKernel(x_tr, landmarks, scale)

    def calcKnmTKnm(self, x_tr, landmarks, scale):
        Knm = self.calcKnm(x_tr, landmarks, scale)
        Knm_T = self.nbTranspose(Knm)
        return np.dot(Knm_T, Knm)

    def calcZm(self, x_tr, y_tr, landmarks, scale):
        Knm = self.calcKnm(x_tr, landmarks, scale)
        Knm_T = self.nbTranspose(Knm)
        return np.dot(Knm_T, y_tr)

    #@nb.jit(nopython=True)
    def nbTranspose(self, X):
        X_T = X.transpose()
        return X_T

class FalkonPrecond():
    """
    This class implements the preconditioner functionality
    for the FALKON algorithm
    """
    def __init__(self, config):
        self.reg_param = float(config['falkon']['reg_param'])
        self.chol_reg = float(config['falkon']['chol_reg'])

        assert isinstance(self.chol_reg, float), f'chol_reg must be float'
        assert isinstance(self.reg_param, float), f'reg_param must be float'

        return

    def create_AandT(self, Kmm, reg_param):
        """
        This functions creates the matrices T, Tt, A, At which are
        used to define the preconditioner B = 1/sqrt(n)*invers(T)*invers(A)
        where T = Cholesky(Kmm) and A = Cholesky((1/m)*T*T.transpose + lambda*I)
        :param Kmm: Matrix
        :return: T, Tt, A, At
        """
        # T matrix and T.transpose matrix
        # self.debug(Kmm)
        m, _ = Kmm.shape
        T = lalg.cholesky(Kmm+self.chol_reg*np.identity(m))
        Tt = self.nbTranspose(T)

        # A matrix and A.transpose matrix
        inter_med = (1/m)*np.dot(T, Tt) + reg_param * np.identity(m)
        A = lalg.cholesky(inter_med)

        return T, A

    def create_chol_solvers(self, Kmm, reg_param):
        """
        :param Kmm: Matrix
        :return: Return handles to lambda functions
        CholR(x): handle to linear solver for T*u= x
        CholRt(x): handle to linear solver for T'*u= x
        CholA(x): handle to linear solver for A*u= x
        CholAt(x): handle to linear solver for A'*u= x
        """

        T, A = self.create_AandT(Kmm, reg_param)

        cholT = self.solve_triangular_system(T, systemType='N')
        cholTt = self.solve_triangular_system(T, systemType='T')
        cholA = self.solve_triangular_system(A, systemType='N')
        cholAt = self.solve_triangular_system(A, systemType='T')

        return cholT, cholTt, cholA, cholAt

    def solve_triangular_system(self, matrix, systemType):
        return lambda x: lalg.solve_triangular(matrix, x, trans=systemType)

    # @nb.jit(nopython=True)
    def nbTranspose(self, X):
        X_T = X.transpose()
        return X_T


class FalkonConjgrad():
    def __init__(self, config):
        self.conj_grad_max_iter = config['falkon']['conj_grad_max_iter']
        self.conj_grad_thr = config['falkon']['conj_grad_thr']

        assert isinstance(self.conj_grad_max_iter, int), f'conj_grad_max_iter must be int'
        assert isinstance(self.conj_grad_thr, float), f'conj_grad_max_iter must be float'

        self.coef_list = []
        self.num_iter = 0
        return

    def conjgrad(self, operatorW, b, x):
        """
        A function to solve W\beta = b, with the
        conjugate gradient method. See also
        wikipedia: https://en.wikipedia.org/wiki/Conjugate_gradient_method

        :param operatorW: an operator matrix, real symmetric positive definite
        :param b: vector, right hand side vector of the system
        :param x: vector, starting guess for \beta
        :return: x which is the estimated value of beta
        """

        if x.size == 0:
            res = b
            if b.ndim > 1:
                print("We have more than 2 dim")
                m_b, dim_b = b.shape
                x = np.zeros((m_b, dim_b))
            else:
                m_b = b.shape
                x = np.zeros((m_b))
        else:
            res = b - operatorW(x)

        p = res
        res_old = np.dot(np.transpose(res), res)
        for i in tqdm(range(int(self.conj_grad_max_iter)), desc='Falkon iterations', colour='green'):
            self.num_iter += 1
            self.coef_list.append(np.reshape(x, (-1, 1)))

            Ap = operatorW(p)
            step = (res_old / np.dot(np.transpose(p), Ap))
            x = x + np.dot(p, step)
            res = res - np.dot(Ap, step)
            res_new = np.dot(np.transpose(res), res)
            if np.sqrt(res_new) < self.conj_grad_thr:
                #print(f"Treshold reached with np.sqrt(res_new): {np.sqrt(res_new)}")
                #print(f"Number of iterations: {self.num_iter}")
                break
            p = res + (res_new / res_old) * p
            res_old = res_new
        #print(f"Number of iterations: {self.num_iter}")
        return x

class FalkonSolver(FalkonPrecond, FalkonConjgrad, FalkonMatrixSystem):
    """This class (fits the kernel model) i.e.
    (solves the for the alphas {\alpha^{(s)}_i}),
    at scale s, in the model function
    f(x) = \Sum_{i=1}^m k_s(x,x_i)\alpha^{(s)}_i,
    where s is the scale
    """
    def __init__(self, config):
        FalkonPrecond.__init__(self, config)
        FalkonConjgrad.__init__(self, config)
        FalkonMatrixSystem.__init__(self, config)

        self.kernelType = config['falkon']['kernelType']
        self.reg_param = config['falkon']['reg_param']
        self.lm_set = False

        assert isinstance(self.kernelType, str), f'kernelType must be str'
        assert isinstance(self.reg_param, float), f'reg_param must be float'

    def fit(self, Kmm, KnmTKnm, Zm, N):
        """
        This function solves the linear system Bt*H*B*\beta = Bt*Knmt*y
        for \beta, and the get \alpha  from \alpha = B\beta.
        :param Kmm: System matrix size, m x m
        :param KnmTKnm: System matrix size, m x m
        :param Zm: System vector size, m
        :param N: Number of training points used to form KnmTKnm and Zm
        :return: alpha, coefficients in the kernel model
        """

        # Get the preconditioner matrices as lambda functions, that solve
        # a triangular linear system
        cholT, cholTt, cholA, cholAt = self.create_chol_solvers(Kmm, self.reg_param)

        # Compute right hand side of W*beta = b, where W = Bt*H*B and b =Bt*Zm
        b = cholAt(cholTt(Zm))

        # Compute the system matrix W as a linear operator that acts on v
        W = lambda v: self.calcW(cholT, cholTt, cholA, cholAt, KnmTKnm, N, v, self.reg_param)

        # Perform conjugate gradient
        _, yd = b.shape
        beta = np.zeros_like(b)

        for i in range(0, yd):
            initialGuessBeta = np.array([])
            beta[:, i] = self.conjgrad(W, b[:, i], initialGuessBeta)
        # Get alpha
        alpha = cholT(cholA(beta))
        return alpha

    def predict(self, X, landmarks, alpha, bw):
        Knm = self.calcKnm(X, landmarks, bw)
        return np.dot(Knm, alpha)

    def get_coef_list(self):
        return self.coef_list.copy()

    def get_kernel_obj(self):
        return self.kernel_obj


