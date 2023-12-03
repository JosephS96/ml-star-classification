import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray):
    sum_sqr = np.sum(np.square(y - x))
    return np.sqrt(sum_sqr)


def manhattan_distance(x, y):
    pass


def cosine_distance(x, y):
    pass
