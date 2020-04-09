import numpy as np


def euclidean_distance(x, y):
    x_sqr_norms = np.array([np.sum(x * x, axis=1)])
    y_sqr_norms = np.array([np.sum(y * y, axis=1)])
    x_y_dot_products = x.dot(y.T)
    dist = x_sqr_norms.T - 2 * x_y_dot_products + y_sqr_norms
    return np.sqrt(dist)
def cosine_distance(x, y):
    x_sqr_norms = np.array([np.linalg.norm(x, axis=1)])
    y_sqr_norms = np.array([np.linalg.norm(y, axis=1)])
    x_y_dot_products = x.dot(y.T)
    cos_dist = x_y_dot_products / x_sqr_norms.T /y_sqr_norms
    return 1 - cos_dist