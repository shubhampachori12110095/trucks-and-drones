'''
'''

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
            'UV_inside_MV': [],
            'UV_outside': [],
            }

        self.base_groups_nodes = {
            'depot': [],
            'customer': [],
            }

        self.current_nodes = []


    def add_restriction(obj, name, base_group=None):
        self.trace_restr_dict[name] = obj

        if base_group is not None:
            base_groups_restr[base_group].append(name)

        #if isinstance(group_list, list):
            #[temp_db.groups_dict[group].append(name) for group in group_list]

    def add_vehicle(obj, name, base_group=None):
        self.trace_vehicle_dict[name] = obj

        if base_group is not None:
            base_groups_vehicles[base_group].append(name)

    
    def add_node(obj, name, base_group=None):
        self.base_groups_nodes[name] = obj

        if base_group is not None:
            base_groups_restr[base_group].append(name)



