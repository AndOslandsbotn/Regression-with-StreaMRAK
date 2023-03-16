from time import perf_counter
import numpy as np
import json
from sklearn.model_selection import train_test_split

def generate_toy_data(seed):
    sigma1 = 0.1
    sigma2 = 0.5
    center1 = 0.1
    center2 = 0.5
    size = 10000

    #####################################
    # Create dummy data
    data = {}
    np.random.seed(seed)
    x = np.random.uniform(low=0, high=1, size=size)
    data['x'] = np.random.uniform(low=0, high=1, size=size).reshape(-1, 1)
    data['y'] = np.exp(-(center1-data['x'])**2/sigma1**2) + np.exp(-(center2-data['x'])**2/sigma2**2)

    xtr, xval, ytr, yval = train_test_split(data['x'], data['y'], test_size=0.3, random_state=42)

    training_data = {}
    training_data['x'] = xtr
    training_data['y'] = ytr

    validation_data = {}
    validation_data['x'] = xval
    validation_data['y'] = yval
    #####################################
    return training_data, validation_data

def _dist(p1, p2):
    """
    :param p1: 1-dim np array
    :param p2: 1-dim np array
    :return: distance between p1 and p2
    """
    return np.sqrt(np.sum((p1-p2)**2, axis=0))

def load_json_data(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        time = t2-t1
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result, time
    return wrap_func


def list_to_ndarray(model):
    model_new = {}
    for key in model:
        lm = np.array(model[key]['lm'])
        bw = model[key]['bw']
        coef = np.array(model[key]['coef'])
        model_new[key] = (lm, bw, coef)
    return model_new

def ndarray_to_list(model):
    model_new = {}
    for key in model:
        lm = model[key][0]
        bw = model[key][1]
        coef = model[key][2]

        model_new[key] = {}
        model_new[key]['lm'] = lm.tolist()
        model_new[key]['bw'] = bw
        model_new[key]['coef'] = coef.tolist()
    return model_new
