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
    return np.array([np.random.randint(0,grid[0]+1), np.random.randint(0,grid[1]+1)])


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

        self.key_groups_dict = {
            'coordinates' : ['v_coord','c_coord','d_coord'],
            'binary'      : ['v_free','v_stuck','v_loaded','v_type','v_loadable'],
            'values'      : ['battery','v_range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
            'vehicles'    : ['v_coord','battery','v_range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','v_free','v_stuck','v_loaded','v_type','v_loadable'],
            'customers'   : ['c_coord','demand'],
            'depots'      : ['d_coord','stock'],
            'restrictions': ['battery','v_range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
            'action_signals': ['cargo_loss','v_free','compare_coord','free_to_travel','unloading_v','free_to_unload_v','free_to_be_loaded_v','free_to_load_v','free_to_unload_cargo','free_to_load_cargo'],
            'restr_signals': [],
        }

        # Init number of objects
        self.num_vehicles  = 0
        self.num_nodes     = 0
        self.num_customers = 0
        self.num_depots    = 0

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
        }

        self.total_time = 0

    def prep_max_min(self, name, max_restr, min_restr, rate):

        append_to_array(self.min_max_dict, name, [max_restr, min_restr])
        append_to_array(self.min_max_dict, 'max_'+name, max_restr)
        append_to_array(self.min_max_dict, 'min_'+name, min_restr)
        append_to_array(self.min_max_dict, 'rate_'+name, rate)

        #print(self.min_max_dict)

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
        insert_at_array(self.constants_dict, 'min_'+name, restr_obj.min_restr, list_index, num_objs)
        insert_at_array(self.constants_dict, 'init_'+name, restr_obj.init_value, list_index, num_objs)
        insert_at_array(self.constants_dict, 'rate_'+name, restr_obj.rate, list_index, num_objs)
        
        # Signals at Signals Dict:
        insert_at_array(self.signals_dict, 'signal_'+name, 0, list_index, num_objs)

    def add_node(self, node, n_index, n_type):

        # Object at Base Group:
        insert_at_list(self.base_groups, 'nodes', node, n_index, self.num_nodes)
        
        # Object at Base Group:
        insert_at_list(self.base_groups, 'nodes', node, n_index, self.num_nodes)
        
        # Variables at Status Dict:
        insert_at_coord(self.status_dict, 'n_coord', random_coordinates(self.grid), n_index, self.num_nodes)

        # Constants at Constants Dict:
        insert_at_array(self.constants_dict, 'n_type', n_type, n_index, self.num_nodes)

        if node.n_name == 'depot':
            self.d_indices.append(n_index)
        elif node.n_name == 'customer':
            self.c_indices.append(n_index)

    def add_vehicle(self, vehicle, v_index, v_type):

        # Object at Base Group:
        insert_at_list(self.base_groups, 'vehicles', vehicle, v_index, self.num_vehicles)

        # Variables at Status Dict:
        insert_at_array(self.status_dict, 'v_free', 1, v_index, self.num_vehicles)
        insert_at_coord(self.status_dict, 'v_coord', random.sample(list(self.depots(self.status_dict['n_coord'])[0]), 1)[0], v_index, self.num_vehicles)
        
        # Constants at Constants Dict:
        insert_at_array(self.constants_dict, 'v_range_type', ['simple', 'battery'].index(vehicle.range_type), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_travel_type', ['street', 'arial'].index(vehicle.travel_type), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_cargo_type', ['standard', 'standard+extra', 'standard+including'].index(vehicle.cargo_type), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_is_truck', int(vehicle.is_truck), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_loadable', int(vehicle.v_loadable), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_weight', int(vehicle.v_weight), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_type', v_type, v_index, self.num_vehicles)

        self.v_indices.append(v_index)

    def reset_db(self):    

        for key in self.min_max_dict.keys():
            self.min_max_dict[key] = np.nan_to_num(self.min_max_dict[key].astype(float))
            self.min_max_dict[key] = np.array([np.min(self.min_max_dict[key]), np.max(self.min_max_dict[key])])

        for key in self.key_groups_dict['action_signals']: self.signals_dict[key] = np.zeros((self.num_vehicles))

        # Reset visited coordinates
        self.past_coord_not_transportable_v = [[] for v in self.base_groups['vehicles'] if not v.v_loadable] ##### ergänze bei vehicles
        self.past_coord_transportable_v     = [[] for v in self.base_groups['vehicles'] if v.v_loadable]     ##### ergänze bei vehicles

        self.status_dict['n_waiting'] = np.zeros((self.num_nodes))
        self.status_dict['v_to_n'] = np.zeros((self.num_vehicles))
        self.status_dict['v_stuck'] = np.zeros((self.num_vehicles))
        self.status_dict['v_dest'] = np.copy(self.status_dict['v_coord'])

        self.cur_v_index = 0
        self.cur_time_frame = 0
        self.actions_list = [[] for i in range(self.num_vehicles)]
        self.v_transporting_v = [[] for i in range(self.num_vehicles)]
        self.time_till_fin = np.zeros((self.num_vehicles))
        self.time_till_fin.fill(None)

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
