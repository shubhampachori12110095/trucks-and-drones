'''
'''

def lookup_db(db_dict, name_list):
    obj_list = []
    return [obj_list.append(db_dict[startswith(name)]) for name in name_list]


class TempDatabase:

    def __init__(self, grid=[10,10]):

        # Grid by x and y size
        self.grid = grid

        # Dict where restriction objects live
        self.base_groups_restr = {
            'battery': [],
            'range': [],
            'cargo': [],
            'cargo_rate': [],
            'cargo_UV': [],
            'cargo_UV_rate': [],
            'stock': [],
            'demand': [],
            }

        # Dict where vehicle and node objects live
        self.base_groups = {
            'vehicles': [],
            'nodes': [],
            }

        # transporter name as key to look up list of loaded vehicles
        self.v_transporting_v = {}

        # List of vehicle names that aren't transported:
        self.free_vehicles = []

        # Current Node Coordinates
        self.cur_coord_nodes = []

        # Current transportable vehicle Coordinates
        self.cur_coord_transportable_v = []

        # Current NOT transportable vehicle Coordinates
        self.cur_coord_not_transportable_v = []

        # Current times till vehicles reach their destination ##### ergänze bei vehicles
        self.times_till_destination = []

    def reset_db(self):
        # Calculate number of vehicles
        self.num_vehicles = len(self.base_groups['vehicles'])
        # Claculate number of nodes
        self.num_nodes    = len(self.base_groups['nodes'])

        # Reset visited coordinates
        self.past_coord_not_transportable_v = [[] for v in self.base_groups['vehicles'] if not v.loadable] ##### ergänze bei vehicles
        self.past_coord_transportable_v     = [[] for v in self.base_groups['vehicles'] if v.loadable]     ##### ergänze bei vehicles


    def init_step(self):

        zero_list_v = [0 for i in range(self.num_vehicles)]
        self.action_signal = {
            'compare_coord'        : zero_list_v, # Deviation of chosen coordinates and coordinates of chosen nodes
            'free_to_travel'       : zero_list_v, # Indicates if chosen vehicle was able to move (or is currently transported)
            'unloading_v'          : zero_list_v, # Deviation of chosen number of vehicles to unload vs the actual vehicles that could be unloaded
            'free_to_unload_v'     : zero_list_v, # Indicates if chosen vehicle was able to unload vehicles (or is currently transported)
            'free_to_be_loaded_v'  : zero_list_v, # Indicates if vehicle to be loaded was actually loaded
            'free_to_load_v'       : zero_list_v, # Indicates if chosen vehicle was able to load a vehicle
            'free_to_unload_cargo' : zero_list_v, # Indicates if chosen vehicle was able to unload cargo (or is currently transported)
            'free_to_load_cargo'   : zero_list_v, # Indicates if chosen vehicle was able to load cargo (or is currently transported)
            }

        [vehicle_obj.cargo_obj.cargo_per_step.reset() for vehicle_obj in self.base_groups['vehicles']]
        [vehicle_obj.cargo_obj.cargo_UV_per_step.reset() for vehicle_obj in self.base_groups['vehicles']]


    def finish_step(self):

        # create dict for restriction signals
        self.restriction_signals = {}
        for key in self.base_groups_restr.keys()
            self.restriction_signals[key] = [elem.cur_signal for elem in self.base_groups_restr[key]]



    def add_restriction(obj, name, base_group=None):
        #self.trace_restr_dict[name] = obj

        if base_group is not None:
            base_groups_restr[base_group].append(obj)

        #if isinstance(group_list, list):
            #[temp_db.groups_dict[group].append(name) for group in group_list]

    def add_vehicle(obj, name, base_group=None):
        #self.trace_vehicle_dict[name] = obj

        if base_group is not None:
            base_groups_vehicles[base_group].append(obj)

    
    def add_node(obj, name, base_group=None):
        #self.base_groups_nodes[name] = obj

        if base_group is not None:
            base_groups_restr[base_group].append(obj)



