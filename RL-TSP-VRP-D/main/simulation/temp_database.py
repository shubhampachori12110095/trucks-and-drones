'''
'''
import random
import numpy as np

'''
def lookup_db(db_dict, name_list):
    obj_list = []
    return [obj_list.append(db_dict[i]) for name in name_list]
'''

def append_to(dict_var, key, value):
    if any(key in elem for elem in dict_var):
        dict_var[key].append(value)
    else:
        dict_var[key] = [value]

def random_coordinates(grid):
    return np.array([np.random.randint(0,grid[0]+1), np.random.randint(0,grid[1]+1)])


class TempDatabase:

    def __init__(self, grid):

        # Grid by x and y size
        self.grid = grid        

        # List of vehicle names that aren't transported:
        #self.free_vehicles = []

        # Current Node Coordinates
        #self.cur_coord_nodes = []

        # Current transportable vehicle Coordinates
        #self.cur_coord_transportable_v = []

        # Current NOT transportable vehicle Coordinates
        self.cur_coord_not_transportable_v = []


        self.max_values_dict = {
            'battery': 100,
            'range': None,
            'cargo': 10,
            'cargo_rate': 1,
            'cargo_UV': 1,
            'cargo_UV_rate': 1,
            'stock': None,
            'demand': 10,
            'v_free': 1,
            'v_stuck': 1,
            'v_loaded': 1,
            'v_type': 1,
            'v_loadable': 1,
            'speed'      : 1,
            'travel_type': 1,
            'range_type' : 1,
        }

        self.min_values_dict = {
            'battery': 0,
            'range': 0,
            'cargo': 0,
            'cargo_rate': 0,
            'cargo_UV': 0,
            'cargo_UV_rate': 0,
            'stock': 0,
            'demand': 0,
            'v_free': 0,
            'v_stuck': 0,
            'v_loaded': 0,
            'v_type': 0,
            'v_loadable': 0,
            'speed'      : 0,
            'travel_type': 0,
            'range_type' : 0,
        }

        self.outputs_max = {
            'load': 10,
            'unload': 10,
            'v_load': 1,
            'v_unload': 1,
        }

        self.key_groups_dict = {
            'coordinates' : ['v_coord','c_coord','d_coord'],
            'binary'      : ['v_free','v_stuck','v_loaded','v_type','v_loadable'],
            'values'      : ['battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
            'vehicles'    : ['v_coord','battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','v_free','v_stuck','v_loaded','v_type','v_loadable'],
            'customers'   : ['c_coord','demand'],
            'depots'      : ['d_coord','stock'],
            'restrictions': ['battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand']
        }


    def init_db(self):

        # Dict where vehicle and node objects live
        self.base_groups = {
            'vehicles' : [],
            'nodes'    : [],
            'depots'   : [],
            'customers': [],
        }

        # Dict with current values of restriction objects
        self.restr_dict = {
            'battery'      : [],
            'range'        : [],
            'cargo'        : [],
            'cargo_rate'   : [],
            'cargo_UV'     : [],
            'cargo_UV_rate': [],
            'stock'        : [],
            'demand'       : [],
        }
        
        self.status_dict = {
            # Vehicles:
            'v_coord'   : [], # list of current vehicle coordinates
            'v_dest': [],
            'time_to_dest': [],
            'v_free'    : [], # list of zeros and ones to indicate if vehicle is free to move
            'v_stuck'   : [], # list of zeros and ones to indicate if vehicle range is depleted (and not at depot or transporter)
            'v_loaded'  : [], # list of zeros and ones to indicate if vehicle is currently transported
            'v_type'    : [], # list of zeros and ones to indicate if vehicle is a transporter
            'v_loadable': [], # list of zeros and ones to indicate if vehicle can be transported

            'speed'      : [],
            'travel_type': [],
            'range_type' : [],

            # Nodes
            'c_coord': [], # list of current customer coordinates
            'd_coord': [], # list of current depot coordinates
            'c_waiting': [],
            'v_to_n': [],
            'v_to_n_index': [],

            # Restrictions
            'battery'     : [],
            'range'       : [],
            'cargo'       : [],
            'cargo_rate'  : [],
            'cargo_v'     : [],
            'cargo_v_rate': [],
            'stock'       : [],
            'demand'      : [],

            'max_battery'     : [],
            'max_range'       : [],
            'max_cargo'       : [],
            'max_cargo_rate'  : [],
            'max_cargo_v'     : [],
            'max_cargo_v_rate': [],
            'max_stock'       : [],
            'max_demand'      : [],

            'min_battery'     : [],
            'min_range'       : [],
            'min_cargo'       : [],
            'min_cargo_rate'  : [],
            'min_cargo_v'     : [],
            'min_cargo_v_rate': [],
            'min_stock'       : [],
            'min_demand'      : [],

            'init_battery'     : [],
            'init_range'       : [],
            'init_cargo'       : [],
            'init_cargo_rate'  : [],
            'init_cargo_v'     : [],
            'init_cargo_v_rate': [],
            'init_stock'       : [],
            'init_demand'      : [],

            'signal_battery'     : [],
            'signal_range'       : [],
            'signal_cargo'       : [],
            'signal_cargo_rate'  : [],
            'signal_cargo_v'     : [],
            'signal_cargo_v_rate': [],
            'signal_stock'       : [],
            'signal_demand'      : [],
        }

    def reset_db(self):    

        # Calculate numbers
        self.num_vehicles  = len(self.base_groups['vehicles'])
        self.num_nodes     = len(self.base_groups['nodes'])
        self.num_customers = len(self.status_dict['demand'])
        self.num_depots    = len(self.status_dict['stock'])

        # transporter name as key to look up list of loaded vehicles
        self.v_transporting_v = [[] for i in range(self.num_vehicles)]

        # Current times till vehicles reach their destination
        self.times_till_destination = [0  for i in range(self.num_vehicles)]
        

        self.visited = [[] for i in range(self.num_vehicles)]

        # sollte vlt bei erstellungsprozess lieber erfolgen?
        #self.status_dict['d_coord'] = [random_coordinates(self.grid) for i in range(self.num_depots)]
        #self.status_dict['c_coord'] = [random_coordinates(self.grid) for i in range(self.num_customers)]

        #trucks = [random.choice(self.status_dict['d_coord']) for i in range(self.num_vehicles) if self.status_dict['v_type'][i] == 1]
        #drones = [random.choice(trucks) for i in range(self.num_vehicles) if self.status_dict['v_type'][i] == 0]

        #self.status_dict['v_coord'] = trucks+drones


        # Reset visited coordinates
        self.past_coord_not_transportable_v = [[] for v in self.base_groups['vehicles'] if not v.v_loadable] ##### ergänze bei vehicles
        self.past_coord_transportable_v     = [[] for v in self.base_groups['vehicles'] if v.v_loadable]     ##### ergänze bei vehicles

        ############################### QUICK FIX !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.status_dict['c_waiting'] = [0  for i in range(self.num_customers)]

        for key in self.status_dict.keys():
            if self.status_dict[key]==[]:
                self.status_dict[key] = [0  for i in range(self.num_vehicles)]

        self.cur_v_index = 0


    def init_step(self):

        zero_list_v = [0 for i in range(self.num_vehicles)]
        self.action_signal = {
            'v_free'               : zero_list_v,
            'compare_coord'        : zero_list_v, # Deviation of chosen coordinates and coordinates of chosen nodes
            'free_to_travel'       : zero_list_v, # Indicates if chosen vehicle was able to move (or is currently transported)
            'unloading_v'          : zero_list_v, # Deviation of chosen number of vehicles to unload vs the actual vehicles that could be unloaded
            'free_to_unload_v'     : zero_list_v, # Indicates if chosen vehicle was able to unload vehicles (or is currently transported)
            'free_to_be_loaded_v'  : zero_list_v, # Indicates if vehicle to be loaded was actually loaded
            'free_to_load_v'       : zero_list_v, # Indicates if chosen vehicle was able to load a vehicle
            'free_to_unload_cargo' : zero_list_v, # Indicates if chosen vehicle was able to unload cargo (or is currently transported)
            'free_to_load_cargo'   : zero_list_v, # Indicates if chosen vehicle was able to load cargo (or is currently transported)
            }

        [vehicle_obj.cargo_obj.cargo_rate.reset() for vehicle_obj in self.base_groups['vehicles']]
        [vehicle_obj.cargo_obj.cargo_UV_rate.reset() for vehicle_obj in self.base_groups['vehicles']]


    def finish_step(self):

        # create dict for restriction signals
        self.restriction_signals = {}
        for key in self.restr_dict.keys():
            self.restriction_signals[key] = [elem.cur_signal for elem in self.restr_dict[key]]


    def add_restriction(self, name, max_restr, min_restr, init_value):
        append_to(self.status_dict, name, init_value)
        append_to(self.status_dict, 'max_'+name, max_restr)
        append_to(self.status_dict, 'min_'+name, min_restr)
        append_to(self.status_dict, 'init_'+name, init_value)
        append_to(self.status_dict, 'signal_'+name, 0)


    def add_vehicle(self, vehicle, travel_type, range_type, speed):
        append_to(self.base_groups, 'vehicles', vehicle)
        
        append_to(self.status_dict, 'travel_type', travel_type)
        append_to(self.status_dict, 'range_type', range_type)
        append_to(self.status_dict, 'speed', speed)
        
        append_to(self.status_dict, 'v_free', 1)
        append_to(self.status_dict, 'v_type', int(vehicle.v_type))
        append_to(self.status_dict, 'v_loadable', int(vehicle.v_loadable))

        append_to(self.status_dict, 'v_coord', random.choice(self.status_dict['d_coord']))


    
    def add_depot(self, depot):
        append_to(self.base_groups, 'depots', depot)
        append_to(self.base_groups, 'nodes', depot)

        append_to(self.status_dict, 'd_coord', random_coordinates(self.grid))


    def add_customer(self, customer):
        append_to(self.base_groups, 'customers', customer)
        append_to(self.base_groups, 'nodes', customer)

        append_to(self.status_dict, 'c_coord', random_coordinates(self.grid))


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


