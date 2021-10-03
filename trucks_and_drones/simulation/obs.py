import numpy as np
from gym import spaces
from scipy.spatial.distance import cdist


def normalize_safe(vals, lows, highs, scale=0.1):
    for i in range(len(vals)):
        if highs[i] is None or (highs[i] - lows[i]) == 0:
            vals[i] = np.tanh(vals[i]*scale)
        else:
            vals[i] = (vals[i] - lows[i]) / (highs[i] - lows[i])
    return vals

def tanh_normalize(val, scale=0.1):
    return  np.tanh(val*scale)


class CustomObs:

    def __init__(self, temp_db):

        self.temp_db = temp_db

    def reset(self):
        pass

    def obs_space(self):
        pass

    def observe_state(self):
        pass

    def check_if_coord(self, key):
        return key in {'v_coord', 'v_dest', 'c_coord', 'd_coord'}

    def count_num_inputs(self, keys):
        pass

    def standardize_array(self, array):
        pass

    def tanh_scaler(self, val, scale=0.1):
        return np.tanh(val * scale)

    def min_max_scaler(self, val, val_low, val_high, min_scale=0, max_scale=1):
        val_std = (val - val_low) / (val_high - val_low)
        return val_std * (max_scale - min_scale) + min_scale

    def calc_euclidean_distance(self, nodes, start_node=None):
        if start_node is None:
            return cdist(nodes, nodes, metric='euclidean')

        if start_node.ndim == 1 and nodes.ndim > 1:
            start_node = np.expand_dims(start_node, axis=0)
        elif start_node.ndim > 1 and nodes.ndim == 1:
            nodes = np.expand_dims(nodes, axis=0)

        return cdist(start_node, nodes, metric='euclidean')

    def calc_manhattan_distance(self, nodes, start_node=None):
        if start_node is None:
            return cdist(nodes, nodes, metric='cityblock')

        if start_node.ndim == 1:
            start_node = np.expand_dims(start_node, axis=0)
        elif start_node.ndim > 1 and nodes.ndim == 1:
            nodes = np.expand_dims(nodes, axis=0)

        return cdist(start_node, nodes, metric='cityblock')

    def euclidean_distances_from_vehicle(self, vehicle_idx, consider_speed=True):
        coord = self.temp_db.status_dict['v_coord'][vehicle_idx]
        distances = self.calc_euclidean_distance(coord, self.temp_db.status_dict['n_coord'])
        if consider_speed:
            distances = distances * self.temp_db('vehicles', vehicle_idx).speed
        return distances

    def manhatten_distances_from_vehicle(self, vehicle_idx, consider_speed=True):
        coord = self.temp_db.status_dict['v_coord'][vehicle_idx]
        distances = self.calc_manhattan_distance(coord, self.temp_db.status_dict['n_coord'])
        if consider_speed:
            distances = distances * self.temp_db('vehicles', vehicle_idx).speed
        return distances

    def coord_to_contin(self, key, set_coord_with_no_demand_to_zero=False):
        ''' Normalizes list of Coordinates and optionally sets customer coord to [0,0] if no demand'''
        coord_list = self.temp_db.get_val(key)

        array_x = np.array([elem[0] / self.temp_db.grid[0] for elem in coord_list])
        array_y = np.array([elem[1] / self.temp_db.grid[1] for elem in coord_list])

        if key == 'c_coord' and set_coord_with_no_demand_to_zero:
            for i in range(self.temp_db.num_customers):
                if self.temp_db.status_dict['n_items'][i + self.temp_db.num_depots] == 0:
                    array_x[i] = 0.0
                    array_y[i] = 0.0

        return np.nan_to_num(np.append(array_x, array_y))

    def coord_to_discrete(self, key):
        ''' Converts list of Coordinates to discrete'''
        coord_list = self.temp_db.get_val(key)

        array_x = np.zeros((len(coord_list), self.temp_db.grid[0] + 1))
        array_y = np.zeros((len(coord_list), self.temp_db.grid[1] + 1))

        array_x[np.arange(len(coord_list)), np.array([int(elem[0]) for elem in coord_list])] = 1
        array_y[np.arange(len(coord_list)), np.array([int(elem[1]) for elem in coord_list])] = 1

        return np.nan_to_num(np.concatenate((array_x, array_y), axis=1))

    def value_to_contin(self, key):
        ''' Normalizes list of Values'''

        max_value = self.temp_db.min_max_dict[key][1]
        min_value = self.temp_db.min_max_dict[key][0]
        value_list = self.temp_db.get_val(key)

        if max_value is None:
            return np.ones((len(value_list)))
        if (max_value - min_value) == 0:
            return np.zeros((len(value_list)))
        return np.nan_to_num((np.array(value_list) - min_value) / (max_value - min_value))

    def binary_to_discrete(self, key):
        ''' Converts list of binary values to discrete'''
        binary_list = self.temp_db.get_val(key)

        array_binary = np.zeros((len(binary_list), 2))
        array_binary[np.arange(len(binary_list)), np.array(binary_list, dtype=np.int)] = 1

        return np.nan_to_num(array_binary)

    def value_to_discrete(self, key, discrete_dims):
        ''' Converts list of Values to discrete'''
        value_list = self.temp_db.get_val(key)

        print(value_list)
        print(self.temp_db.min_max_dict[key][1])
        print(self.temp_db.min_max_dict[key][0])

        array_value = np.zeros((len(value_list), discrete_dims))

        max_value = self.temp_db.min_max_dict[key][1]
        min_value = self.temp_db.min_max_dict[key][0]

        if max_value is None:
            values = np.ones((len(value_list))) * (discrete_dims - 1)
        else:
            values = (np.array(value_list) - min_value) // (max_value - min_value) * (discrete_dims - 1)

        array_value[np.arange(len(value_list)), values] = 1

        return np.nan_to_num(array_value)


class SimpleObs(CustomObs):

    def __init__(self, temp_db, keys=None):
        super(SimpleObs, self).__init__(temp_db)

        if keys is None:
            keys = ['v_coord', 'v_dest', 'c_coord', 'd_coord', 'demand','v_items']
        self.keys = keys

        self.num_inputs = 64 #self.count_num_inputs(keys)

    def obs_space(self):
        return spaces.Box(low=0, high=1, shape=(self.num_inputs,))

    def observe_state(self):

        inputs = np.array([])
        for key in self.keys:
            if self.check_if_coord(key):
                inputs = np.append(inputs, self.coord_to_contin(key, True))
            else:
                inputs = np.append(inputs, self.value_to_contin(key))

        distances = np.array([])
        distances = np.append(distances, self.manhatten_distances_from_vehicle(0, consider_speed=False))
        distances = np.append(distances, self.euclidean_distances_from_vehicle(1, consider_speed=False))

        distances = self.min_max_scaler(distances, val_low=0, val_high=np.max(self.temp_db.grid))

        return np.append(inputs,distances)


class DiscreteObserver:

    def __init__(
            self,
            database,
            keys: (str, list),
            bins: int = None,
    ):

        self.temp_db = database
        self.keys = keys

        self.num_obs = self.temp_db.count_elements(self.keys)
        self.lows = self.temp_db.minimas(self.keys)
        self.highs = self.temp_db.maximas(self.keys)

        if bins is None:
            self.bins = np.nanmax(self.highs-self.lows)
        else:
            self.bins = bins

        if self.bins == np.nan or self.bins is None:
            raise Exception('Failed to calculate number of bins for keys:'+str(keys))

    def __call__(self):

        vals = np.ravel(self.temp_db(self.keys))
        vals = normalize_safe(vals, self.lows, self.highs)

        if isinstance(vals, str):
            return self._get_discrete(vals)

        discrete_array = np.array([])
        for val in vals:
            np.append(discrete_array, self._get_discrete(val))
        return discrete_array

    def _get_discrete(self, val):
        val = int(val * self.bins)
        placeholder_array = np.zeros((self.bins))
        placeholder_array[val] = 1

    @property
    def gym_space(self):
        return spaces.Discrete(self.bins*self.num_obs)



class ContinObserver:

    def __init__(
            self,
            database,
            keys: (str, list),
            normalize: bool = False,
    ):

        self.temp_db = database
        self.keys = keys
        self.normalize = normalize

        self.num_obs = self.temp_db.count_elements(self.keys)
        self.lows = self.temp_db.minimas(self.keys)
        self.highs = self.temp_db.maximas(self.keys)

    def __call__(self):
        vals = np.ravel(self.temp_db(self.keys))
        if self.normalize:
            vals = normalize_safe(vals, self.lows, self.highs)
        return vals

    @property
    def gym_space(self):
        if self.normalize:
            return spaces.Box(low=np.zeros((self.num_obs)), high=np.ones((self.num_obs)),  dtype=np.float32)
        return spaces.Box(low=self.lows, high=self.highs,  dtype=np.float32)
