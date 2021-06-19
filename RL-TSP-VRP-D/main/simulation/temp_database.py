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


class TempDatabase:

    def __init__(self, grid):

        # Grid by x and y size
        self.grid = grid

        self.key_groups_dict = {
            'coordinates' : ['v_coord','c_coord','d_coord'],
            'binary'      : ['v_free','v_stuck','v_loaded','v_type','v_loadable'],
            'values'      : ['battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
            'vehicles'    : ['v_coord','battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','v_free','v_stuck','v_loaded','v_type','v_loadable'],
            'customers'   : ['c_coord','demand'],
            'depots'      : ['d_coord','stock'],
            'restrictions': ['battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
            'action_signals': ['v_free','compare_coord','free_to_travel','unloading_v','free_to_unload_v','free_to_be_loaded_v','free_to_load_v','free_to_unload_cargo','free_to_load_cargo'],
            'restr_signals': [],
        }

        # Init number of objects
        self.num_vehicles  = 0
        self.num_nodes     = 0
        self.num_customers = 0
        self.num_depots    = 0

        self.max_min_dict = {
            'x_coord': np.array([0, self.grid[0]])
            'y_coord': np.array([0, self.grid[1]])
            'loadable': np.array([0,1]),
            'is_truck': np.array([0,1]),
            'range_type': np.array([0,1]),
            'travel_type': np.array([0,1]),
            'cargo_type': np.array([0,2]),
        }



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


    def reset_db(self):    

        for key in self.max_min_dict.keys(): self.max_min_dict[key] = np.array([np.min(self.max_min_dict[key]), np.max(self.max_min_dict[key])])
        for key in self.key_groups_dict[action_signals]: self.signals_dict[key] = np.zeros((self.num_vehicles))

        # transporter name as key to look up list of loaded vehicles
        self.v_transporting_v = [[] for i in range(self.num_vehicles)]

        # Current times till vehicles reach their destination
        self.times_till_destination = np.zeros((self.num_vehicles))
        

        self.visited = [[] for i in range(self.num_vehicles)]

        # Reset visited coordinates
        self.past_coord_not_transportable_v = [[] for v in self.base_groups['vehicles'] if not v.v_loadable] ##### ergänze bei vehicles
        self.past_coord_transportable_v     = [[] for v in self.base_groups['vehicles'] if v.v_loadable]     ##### ergänze bei vehicles

        ############################### QUICK FIX !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.status_dict['c_waiting'] = np.zeros((self.num_customers))

        for key in self.status_dict.keys():
            if self.status_dict[key]==[]:
                self.status_dict[key] = np.zeros((self.num_vehicles))

        self.cur_v_index = 0


    def init_step(self):

        for key in self.signals_dict.keys(): self.signals_dict[key] = self.signals_dict[key].fill(0)


    def finish_step(self):

        # create dict for restriction signals
        self.restriction_signals = {}
        for key in self.restr_dict.keys():
            self.restriction_signals[key] = [elem.cur_signal for elem in self.restr_dict[key]]

    def prep_max_min(self, name, max_restr, min_restr, rate):

        append_to_array(self.max_min_dict, name, [max_restr, min_restr])
        append_to_array(self.max_min_dict, 'max_'+name, max_restr)
        append_to_array(self.max_min_dict, 'min_'+name, min_restr)
        append_to_array(self.max_min_dict, 'rate_'+name, rate)


    def add_restriction(self, restr_obj, name, max_restr, min_restr, init_value, rate, list_index, index_type):

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
        insert_at_list(self.base_groups, 'nodes', node, self.n_index, self.num_nodes)
        
        # Variables at Status Dict:
        insert_at_coord(self.status_dict, 'n_coord', random_coordinates(self.grid), self.n_index, self.num_nodes)

        # Constants at Constants Dict:
        insert_at_array(self.constants_dict, 'n_type', n_type, n_index, self.num_nodes)

        if node.name == 'depot':
            self.d_indices.append(n_index)
        elif node.name == 'customer':
            self.c_indices.append(n_index)


    def add_vehicle(self, vehicle, v_index, v_type):

        # Object at Base Group:
        insert_at_list(self.base_groups, 'vehicles', vehicle, v_index, self.num_vehicles)

        # Variables at Status Dict:
        insert_at_array(self.status_dict, 'v_free', 1, v_index, self.num_vehicles)
        insert_at_coord(self.status_dict, 'v_coord', random.choice(self.depots(self.status_dict['n_coord'])), v_index, self.num_vehicles)
        
        # Constants at Constants Dict:
        insert_at_array(self.constants_dict, 'v_range_type', ['simple', 'battery'].index(vehicle.range_type), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_travel_type', ['street', 'arial'].index(vehicle.travel_type), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_cargo_type', ['standard', 'standard+extra', 'standard+including'].index(vehicle.cargo_type), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_is_truck', int(vehicle.is_truck), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_loadable', int(vehicle.v_loadable), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_weight', int(vehicle.v_weight), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_type', v_type, v_index, self.num_vehicles)


    def depots(self, array_from_dict, include=None):

        if include is not None:
            indices = set(self.d_indices)
            for elem in include:
                indices = set([i for i in range(len(self.num_nodes)) if elem[0][i] == elem[1]]) & indices
            return array_from_dict[list(indices)]
        return array_from_dict[self.d_indices]


    def customers(self, array_from_dict, include=None):

        if include is not None:
            indices = set(self.c_indices)
            for elem in include:
                indices = set([i for i in range(len(self.num_nodes)) if elem[0][i] == elem[1]]) & indices
            return array_from_dict[list(indices)]
        return array_from_dict[self.c_indices]


    def nearest_neighbour(self, v_index, coord_key, exclude=None):

        v_coord = self.status_dict['v_coord'][v_index]

        if isinstance(coord_key, (list, tuple, np.ndarray)):
            coord_list = []
            for elem in coord_key:
                coord_list += self.status_dict[elem]
        else:
            coord_list = self.status_dict[coord_key]

        compared = [np.sum(np.abs(np.array(elem)-np.array(v_coord))) for elem in coord_list]

        if exclude is not None:
            for i in range(len(compared)):
                for elem in exclude:
                    if self.status_dict[elem[0]][i] == elem[1]:
                        compared[i] = 10000

        return np.argmin(compared)

    def same_coord(self, v_index, coord_index, coord_key):
        print(self.status_dict['v_coord'][v_index], self.status_dict[coord_key][coord_index])
        check = np.sum(np.array(self.status_dict['v_coord'][v_index]) - np.array(self.status_dict[coord_key][coord_index])) == 0
        print(check)
        return check


