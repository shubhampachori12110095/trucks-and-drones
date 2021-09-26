'''
'''
import random
import numpy as np

'''
def lookup_db(db_dict, name_list):
    obj_list = []
    return [obj_list.append(db_dict[i]) for name in name_list]
'''

def random_coordinates(grid):
    return np.array([np.random.randint(1,grid[0]+1), np.random.randint(1,grid[1]+1)])


def insert_at_coord(dict_var, key, value, list_index, num_objs):
    if any(key in elem for elem in dict_var):
        dict_var[key][list_index] = value
    else:
        dict_var[key] = np.zeros((num_objs, 2))
        dict_var[key][list_index] = value


def insert_at_array(dict_var, key, value, list_index, num_objs):
    if any(key in elem for elem in dict_var):
        dict_var[key][list_index] = value
    else:
        dict_var[key] = np.zeros((num_objs))
        dict_var[key][list_index] = value


def insert_at_list(dict_var, key, value, list_index, num_objs):
    if any(key in elem for elem in dict_var):
        dict_var[key][list_index] = value
    else:
        dict_var[key] = [None for i in range(num_objs)]
        dict_var[key][list_index] = value


def append_to_array(dict_var, key, value):
    if any(key in elem for elem in dict_var):
        dict_var[key] = np.append(dict_var[key], value)
    else:
        dict_var[key] = np.array([value])


class BaseTempDatabase:

    def __init__(self, name, grid, signal_list, debug_mode=False):

        self.name = name
        
        # Grid by x and y size
        self.grid = grid

        self.signal_list = signal_list

        self.debug_mode = debug_mode

        status_dict_keys = ['n_items', 'in_time_n_items', 'n_coord', 'v_range', 'in_time_v_range', 'v_items',
                'in_time_v_items', 'v_cargo', 'in_time_v_cargo', 'loaded_v', 'in_time_loaded_v',
                'v_free', 'v_coord', 'n_waiting', 'v_to_n', 'v_stuck', 'v_dest']

        constants_dict_keys = ['max_n_items', 'min_n_items', 'init_n_items', 'rate_n_items', 'n_type',
                               'max_v_range', 'min_v_range', 'init_v_range', 'rate_v_range', 'max_v_items',
                               'min_v_items', 'init_v_items', 'rate_v_items', 'max_v_cargo', 'min_v_cargo',
                               'init_v_cargo', 'rate_v_cargo', 'max_loaded_v', 'min_loaded_v', 'init_loaded_v',
                               'rate_loaded_v', 'v_range_type', 'v_travel_type', 'v_cargo_type',
                               'v_is_truck', 'v_loadable', 'v_weight', 'v_type']

        self.key_groups_dict = {
            'coordinates' : ['v_coord','v_dest','c_coord','d_coord'],
            'binary'      : ['v_free','v_stuck','loaded_v','v_loadable', 'v_is_truck', 'v_range_type', 'v_travel_type',
                             'v_cargo_type'],
            'values'      : ['v_range', 'v_items','v_cargo', 'v_to_n', 'rate_loaded_v','v_weight', 'v_type',
                             'rate_v_cargo', 'rate_v_items', 'rate_v_range', 'demand', 'stock'],
            'vehicles'    : ['v_coord', 'v_range','v_items','v_cargo', 'loaded_v','v_free','v_to_n',
                             'v_stuck', 'v_dest', 'rate_loaded_v', 'v_range_type', 'v_travel_type', 'v_cargo_type',
                             'v_is_truck', 'v_loadable', 'v_weight', 'v_type', 'rate_v_cargo', 'rate_v_items',
                             'rate_v_range'
                             ],
            'customers'   : ['d_coord', 'c_coord','demand'],
            'depots'      : ['d_coord','stock'],
            'restrictions': ['v_range'],
            'action_signals': ['cargo_loss','v_free','compare_coord','free_to_travel','unloading_v','free_to_unload_v','free_to_be_loaded_v','free_to_load_v','free_to_unload_cargo','free_to_load_cargo'],
            'restr_signals': [],
        }

        # Init number of objects
        self.num_vehicles  = 0
        self.num_nodes     = 0
        self.num_customers = 0
        self.num_depots    = 0

        self.bestrafung = 0

    def __call__(self, keys, index=None):
        if isinstance(keys, str):
            return self.lookup_key(keys, index)
        return [self.lookup_key(key, index) for key in keys]

    def lookup_key(self, key, index=None):

        if key in set(self.status_dict.keys()):
            elem = self.status_dict[key]
        elif key in set(self.constants_dict.keys()):
            elem = self.constants_dict[key]
        elif key in set(self.signals_dict.keys()):
            elem = self.signals_dict[key]
        elif key in set(self.key_groups_dict.keys()):
            elem = self.key_groups_dict[key]
        else:
            raise KeyError('Key not found:', key)

        if not index is None:
            return elem[index]
        return elem


    def init_db(self):

        # Dict of vehicle and node objects:
        self.base_groups = {}

        # Dict of restriction objects:
        self.restr_dict = {}
        
        # Dicts of values:
        self.status_dict = {}
        self.constants_dict = {}
        self.signals_dict = {}

        # Init indices:
        self.d_indices = []
        self.c_indices = []
        self.v_indices = []

        # Init visuals:
        self.vehicle_visuals = []
        self.node_visuals = []

        self.min_max_dict = {
            'x_coord': np.array([0, self.grid[0]]),
            'y_coord': np.array([0, self.grid[1]]),
            'loadable': np.array([0,1]),
            'is_truck': np.array([0,1]),
            'range_type': np.array([0,1]),
            'travel_type': np.array([0,1]),
            'cargo_type': np.array([0,2]),
            'v_weight': np.zeros((1)),
        }

        self.total_time = 0
        self.prev_total_time = 0

    def prep_max_min(self, name, max_restr, min_restr, rate):

        append_to_array(self.min_max_dict, name, [max_restr, min_restr])
        append_to_array(self.min_max_dict, 'max_'+name, max_restr)
        append_to_array(self.min_max_dict, 'min_'+name, min_restr)
        append_to_array(self.min_max_dict, 'rate_'+name, rate)

        #print(self.min_max_dict)

    def add_key_to(self, group, key):
        if not any(key in elem for elem in self.key_groups_dict[group]):
            self.key_groups_dict[group].append(key)

    def add_restriction(self, restr_obj, name, list_index, index_type):

        if index_type == 'vehicle':
            num_objs = self.num_vehicles
        else:
            num_objs = self.num_nodes

        # Object at Base Group:
        insert_at_list(self.restr_dict, name, restr_obj, list_index, num_objs)

        # Variables at Status Dict:
        insert_at_array(self.status_dict, name, restr_obj.init_value, list_index, num_objs)
        insert_at_array(self.status_dict, 'in_time_'+name, 0, list_index, num_objs)

        # Constants at Constants Dict:
        insert_at_array(self.constants_dict, 'max_'+name, restr_obj.max_restr, list_index, num_objs)
        self.add_key_to('values', 'max_'+name)

        insert_at_array(self.constants_dict, 'min_'+name, restr_obj.min_restr, list_index, num_objs)
        self.add_key_to('values', 'min_'+name)

        insert_at_array(self.constants_dict, 'init_'+name, restr_obj.init_value, list_index, num_objs)

        insert_at_array(self.constants_dict, 'rate_'+name, restr_obj.rate, list_index, num_objs)
        self.add_key_to('values', 'rate_'+name)

        
        # Signals at Signals Dict:
        insert_at_array(self.signals_dict, 'signal_'+name, 0, list_index, num_objs)

    def add_node(self, node, n_index, n_type):

        # Object at Base Group:
        insert_at_list(self.base_groups, 'nodes', node, n_index, self.num_nodes)
        
        # Object at Base Group:
        insert_at_list(self.base_groups, 'nodes', node, n_index, self.num_nodes)
        
        # Variables at Status Dict:
        insert_at_coord(self.status_dict, 'n_coord', random_coordinates(self.grid), n_index, self.num_nodes)
        self.add_key_to('coordinates', 'n_coord')

        # Constants at Constants Dict:
        insert_at_array(self.constants_dict, 'n_type', n_type, n_index, self.num_nodes)
        self.add_key_to('values', 'n_type')

        if node.n_name == 'depot':
            self.d_indices.append(n_index)
        elif node.n_name == 'customer':
            self.c_indices.append(n_index)

    def add_vehicle(self, vehicle, v_index, v_type):

        # Object at Base Group:
        insert_at_list(self.base_groups, 'vehicles', vehicle, v_index, self.num_vehicles)

        # Variables at Status Dict:
        insert_at_array(self.status_dict, 'v_free', 1, v_index, self.num_vehicles)
        self.add_key_to('binary', 'v_free')
        self.add_key_to('vehicles', 'v_free')

        insert_at_coord(self.status_dict, 'v_coord', random.sample(list(self.depots(self.status_dict['n_coord'])[0]), 1)[0], v_index, self.num_vehicles)
        self.add_key_to('coordinates', 'v_coord')

        # Constants at Constants Dict:
        insert_at_array(self.constants_dict, 'v_range_type', ['simple', 'battery'].index(vehicle.range_type), v_index, self.num_vehicles)
        self.add_key_to('binary', 'v_range_type')
        self.add_key_to('vehicles', 'v_range_type')

        insert_at_array(self.constants_dict, 'v_travel_type', ['street', 'arial'].index(vehicle.travel_type), v_index, self.num_vehicles)
        self.add_key_to('binary', 'v_travel_type')
        self.add_key_to('vehicles', 'v_travel_type')

        insert_at_array(self.constants_dict, 'v_cargo_type', ['standard', 'standard+extra', 'standard+including'].index(vehicle.cargo_type), v_index, self.num_vehicles)
        self.add_key_to('binary', 'v_cargo_type')
        self.add_key_to('vehicles', 'v_cargo_type')

        insert_at_array(self.constants_dict, 'v_is_truck', int(vehicle.is_truck), v_index, self.num_vehicles)
        self.add_key_to('binary', 'v_is_truck')
        self.add_key_to('vehicles', 'v_is_truck')

        insert_at_array(self.constants_dict, 'v_loadable', int(vehicle.v_loadable), v_index, self.num_vehicles)
        self.add_key_to('binary', 'v_loadable')
        self.add_key_to('vehicles', 'v_loadable')

        insert_at_array(self.constants_dict, 'v_weight', int(vehicle.v_weight), v_index, self.num_vehicles)
        self.add_key_to('values', 'v_weight')
        self.add_key_to('vehicles', 'v_weight')

        insert_at_array(self.constants_dict, 'v_type', v_type, v_index, self.num_vehicles)
        self.add_key_to('values', 'v_type')
        self.add_key_to('vehicles', 'v_type')

        self.v_indices.append(v_index)

    def reset_db(self):    

        for key in self.min_max_dict.keys():
            self.min_max_dict[key] = np.nan_to_num(self.min_max_dict[key].astype(float))
            self.min_max_dict[key] = np.array([np.min(self.min_max_dict[key]), np.max(self.min_max_dict[key])])

        self.min_max_dict['stock'] = self.min_max_dict['n_items']
        self.min_max_dict['demand'] = self.min_max_dict['n_items']
        self.min_max_dict['v_to_n'] = np.array([0,self.num_nodes])

        for key in self.key_groups_dict['action_signals']: self.signals_dict[key] = np.zeros((self.num_vehicles))

        # Reset visited coordinates
        self.past_coord_not_transportable_v = [[] for v in self.base_groups['vehicles'] if not v.v_loadable] ##### ergänze bei vehicles
        self.past_coord_transportable_v     = [[] for v in self.base_groups['vehicles'] if v.v_loadable]     ##### ergänze bei vehicles

        self.status_dict['n_waiting'] = np.zeros((self.num_nodes))
        self.status_dict['v_to_n'] = np.zeros((self.num_vehicles))
        self.status_dict['v_stuck'] = np.zeros((self.num_vehicles))
        self.status_dict['v_dest'] = np.copy(self.status_dict['v_coord'])

        self.bestrafung_multiplier = np.ones((self.num_nodes))

        self.cur_v_index = 0
        self.cur_time_frame = 0
        self.actions_list = [[] for i in range(self.num_vehicles)]
        self.v_transporting_v = [[] for i in range(self.num_vehicles)]
        self.time_till_fin = np.zeros((self.num_vehicles))
        self.time_till_fin.fill(None)

        self.done = False

    def init_step(self):

        [self.signals_dict[key].fill(0) for key in self.signals_dict.keys()]

    def finish_step(self):

        # create dict for restriction signals
        '''
        self.restriction_signals = {}
        for key in self.restr_dict.keys():
            self.restriction_signals[key] = [elem.cur_signal for elem in self.restr_dict[key]]
        '''

    def depots(self, array_from_dict, include=None, exclude=None):
        indices = self.find_indices(self.d_indices, self.num_nodes, include, exclude)
        return [array_from_dict[indices], indices]

    def customers(self, array_from_dict, include=None, exclude=None):
        indices = self.find_indices(self.c_indices, self.num_nodes, include, exclude)
        return [array_from_dict[indices], indices]

    def vehicles(self, array_from_dict, include=None, exclude=None):
        indices = self.find_indices(self.v_indices, self.num_vehicles, include, exclude)
        return [array_from_dict[indices], indices]

    def find_indices(self, indices, num_objs, include, exclude):
        indices = set(indices)

        if include is not None:
            for elem in include:
                indices = set([i for i in range(num_objs) if elem[0][i] == elem[1]]) & indices

        if exclude is not None:
            for elem in exclude:
                indices = indices.difference(set([i for i in range(num_objs) if elem[0][i] == elem[1]]))

        return list(indices)

    def nearest_neighbour(self, coord_and_indices):

        v_coord = self.status_dict['v_coord'][self.cur_v_index]
        
        coord = coord_and_indices[0]
        compared = np.sum(np.abs(coord - v_coord), axis=1)

        if compared.size != 0:
            indices = coord_and_indices[1][np.argmin(compared)]
            if isinstance(indices, int):
                return indices
            else:
                return indices[0]
        return None

    def same_coord(self, compare_coord):
        check = np.sum(self.status_dict['v_coord'][self.cur_v_index] - compare_coord) == 0
        return check

    def same_coord_at_destination(self, compare_coord):
        check = np.sum(self.status_dict['v_dest'][self.cur_v_index] - compare_coord) == 0
        return check

    def terminal_state(self):

        if np.sum(self.customers(self.status_dict['n_items'])[0]) == 0:

            d_coord = self.depots(self.status_dict['n_coord'])[0]
            v_coord = self.status_dict['v_coord']

            all_compare = []
            for elem in v_coord:
                compare = np.min(np.abs(d_coord-elem))
                all_compare.append(compare)

            if np.round(np.sum(all_compare), 2) == 0:
                return True
        return False

    def get_val(self, key):
        if key == 'c_coord':
            return self.customers(self.status_dict['n_coord'])[0]
        if key == 'd_coord':
            return self.depots(self.status_dict['n_coord'])[0]
        if key == 'demand':
            return self.customers(self.status_dict['n_items'])[0]
        if key == 'stock':
            return self.depots(self.status_dict['n_items'])[0]
        try:
            return self.status_dict[key]
        except:
            return self.constants_dict[key]


    def possible_nodes(self):

        possible_customer = self.customers(self.status_dict['n_items'], exclude=[[self.status_dict['n_items'], 0]])[1]
        if len(possible_customer) == 0:
            return self.d_indices
        else:
            return possible_customer

    def total_time_delta(self):
        delta = self.total_time - self.prev_total_time
        self.prev_total_time = self.total_time
        return delta
