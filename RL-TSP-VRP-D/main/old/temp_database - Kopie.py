'''
'''

def lookup_db(db_dict, name_list):
    obj_list = []
    return [obj_list.append(db_dict[startswith(name)]) for name in name_list]


class TempDatabase:

    def __init__(self, grid=[10,10]):

        self.grid = grid

        self.trace_restr_dict   = {}
        self.trace_vehicle_dict = {}
        self.trace_node_dict = {}

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

        self.base_groups_vehicles = {
            'MV': [],
            'UV': [],
            #'UV_inside_MV': [],
            #'UV_outside': [],
            }

        self.base_groups_nodes = {
            'depot': [],
            'customer': [],
            }

        self.current_nodes = []

        # MV name as key to look up list of UV names
        self.MV_transporting_UV = {}

        # List of UV name outside:
        self.UV_outside = []


    def add_restriction(obj, name, base_group=None):
        self.trace_restr_dict[name] = obj

        if base_group is not None:
            base_groups_restr[base_group].append(obj)

        #if isinstance(group_list, list):
            #[temp_db.groups_dict[group].append(name) for group in group_list]

    def add_vehicle(obj, name, base_group=None):
        self.trace_vehicle_dict[name] = obj

        if base_group is not None:
            base_groups_vehicles[base_group].append(obj)

    
    def add_node(obj, name, base_group=None):
        self.base_groups_nodes[name] = obj

        if base_group is not None:
            base_groups_restr[base_group].append(obj)



