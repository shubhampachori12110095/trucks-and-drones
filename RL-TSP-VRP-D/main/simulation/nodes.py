'''
Module for Customer and Depot Nodes Creation

Customer Möglichkeiten:
- kommen alle customers aufeinmal oder nach gegebener zeit
- customer haben einen einmaligen demand oder rechargen demand
- (verlieren demand falls nicht gestillt, das heißt sie beziehen über anderen anbieter)
- alle coordinates random oder bekannt

Depot Möglichkeiten:
- depots haben maximales stock limit
- können rechargen

'''
from main.simulation.restrictions import RestrValueObject
from main.simulation.common_sim_func import param_interpret, random_coordinates, max_param_val

'''
NODE PARAMETER
# number of nodes:
num: (int, list, tuple, np.ndarray) = 1,
# node type:
n_name: str = 'depot', # alt: 'depot', 'customer'
# items (stock if node is depot and demand if node is customer):
max_items: (NoneType, int, list, tuple, np.ndarray) = 10,
init_items: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
item_rate: (NoneType, int, list, tuple, np.ndarray) = None,
item_recharge: (NoneType, int, list, tuple, np.ndarray) = 0,
init_items_at_step: (NoneType, int, list, tuple, np.ndarray) = 0,
# visualization:
symbol: (str, NoneType) = 'rectangle', # 'triangle-up', 'triangle-down', 'rectangle'
color: (str, NoneType, list, tuple, np.ndarray) = 'orange',
'''

# Vehicle Class Functions:
# ----------------------------------------------------------------------------------------------------------------

def item_recharge(self, time):


def init_items_at_step(self, time):


# Base Node Class:
# ----------------------------------------------------------------------------------------------------------------

class BaseNodeClass:

    def __init__(self, temp_db, n_index, n_type, n_params):

        # Init:
        self.temp_db = temp_db
        self.n_index = n_index
        self.n_type = n_type

        # Intepret parameter dict:
        for key in n_params.keys(): n_params[key] = param_interpret(n_params[key])

        # Init parameter based on parameter dict
        self.n_name = n_params['n_name']

        # Create items as restricted value:
        self.items = RestrValueObject('items', n_index, temp_db, n_params['max_items'], 0, n_params['init_items'])
        self.item_rate = RestrValueObject('item_rate', n_index, temp_db, n_params['item_rate'], 0, n_params['item_rate'])

        # Init time dependent functions (functions that take the passed time as input):
        self.time_dependent_funcs = []
        # Add item recharge as time dependent function if needed:
        if n_params['item_recharge'] is not None:
            self.time_dependent_funcs.append(item_recharge)
        # Add init items at step as time dependent function if needed:
        if n_params['init_items_at_step'] is not None:
            self.time_dependent_funcs.append(init_items_at_step)

    def step(self, time):

        [func(time) for func in self.time_dependent_funcs]


# Base Node Creator:
# ----------------------------------------------------------------------------------------------------------------

class BaseNodeCreator:

    def __init__(self, temp_db, n_params_list, NodeClass=BaseNodeClass):
        
        self.temp_db = temp_db
        self.n_params_list = n_params_list

        self.temp_db.num_nodes = sum([np.max(n_params['num']) for n_params in n_params_list])
        self.temp_db.num_depots = sum([np.max(n_params['num']) for n_params in n_params_list if n_params['name'] == 'depot'])
        self.temp_db.num_customers = sum([np.max(n_params['num']) for n_params in n_params_list if n_params['name'] == 'customer'])
        self.temp_db.min_max_dict['n_type'] = np.array([0, len(v_params_list) - 1])

    def create(self):

        n_index = 0
        n_type = 0

        for n_params in n_params_list:
            for i in range(n_params['num']):
                node = NodeClass(self.temp_db, n_index, n_type, n_params)
                self.temp_db.add_node(node, n_index, n_type)
                n_index +=1
            n_type += 1

