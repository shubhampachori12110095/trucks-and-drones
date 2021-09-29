import numpy as np
from scipy.spatial.distance import cdist

class BaseDistanceMatrices:

    def __init__(self):
        pass

    def calc_euclidean_distance(self, nodes, start_node=None):
        if start_node is None:
            return cdist(nodes, nodes, metric='euclidean')

        if start_node.ndim == 1:
            start_node = np.expand_dims(start_node, axis=0)

        return cdist(start_node, nodes, metric='euclidean')

    def calc_manhattan_distance(self, nodes, start_node=None):
        if start_node is None:
            return cdist(nodes, nodes, metric='cityblock')

        if start_node.ndim == 1:
            start_node = np.expand_dims(start_node, axis=0)

        return cdist(start_node, nodes, metric='cityblock')


test = BaseDistanceMatrices()

start_node = np.array([0,0])
nodes = np.array([[0,1], [2,1], [1,1], [3,3], [4,1]])
print(test.calc_euclidean_distance(nodes, start_node))
print(test.calc_manhattan_distance(nodes, start_node))
print(test.calc_euclidean_distance(nodes))
print(test.calc_manhattan_distance(nodes))