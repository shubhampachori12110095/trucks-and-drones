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
from restrictions import RestrValueObject
from common_sim_func import param_interpret, random_coordinates, max_param_val


# Parameter:
# ----------------------------------------------------------------------------------------------------------------

def depot_parameter(
        num_depots       = 1,
        max_stock        = None,
        resupply_rate    = 1,
        unlimited_supply = True,
        init_value       = 30,
        signal_list      = [1,1,-1],
        ):
    return {
        'max_stock': max_stock,
        'resupply_rate': resupply_rate,
        'unlimited_supply': unlimited_supply,
        'init_value': init_value,
        'signal_list': signal_list,
        }


def customer_parameter(
        customer_type      = 'static',
        num_customers      = [5,10],
        first_demand_step  = [0,0],
        demand_after_steps = None,
        demand_add         = 1,
        max_demand         = None,
        init_value         = [1,1],
        signal_list        = [1,1,-1],
       ):
    return {
        'customer_type': customer_type,
        'num_customers': num_customers,
        'init_demand': init_demand,
        'first_demand_step': first_demand_step,
        'demand_after_steps': demand_after_steps,
        'demand_add': demand_add,
        'signal_list': signal_list,
        }


# Nodes:
# ----------------------------------------------------------------------------------------------------------------

class NodeCreator:

    def __init__(self, temp_db, customer_parameter, depot_parameter):
        self.temp_db = temp_db

        self.depot_parameter = depot_parameter
        self.num_depots      = depot_parameter['num_depots']
        del self.depot_parameter['num_depots']

        self.customer_parameter = customer_parameter
        self.num_customers      = customer_parameter['num_customers']
        del self.customer_parameter['num_customers']

        self.temp_db.init_nodes(max_param_val(self.num_depots)+max_param_val(self.num_customers))


    def create_depot(self, n_index):
        coordinates  = random_coordinates(self.temp_db.grid)
        customer_obj = BaseCustomer(n_index, self.temp_db, self.customer_parameter, coordinates)
        self.temp_db.add_customer(customer_obj)


    def create_customer(self, n_index):
        coordinates  = random_coordinates(self.temp_db.grid)
        customer_obj = BaseCustomer(n_index, self.temp_db, self.customer_parameter, coordinates)
        self.temp_db.add_customer(customer_obj)

    
    def create_depot(self, n_index):
        coordinates  = random_coordinates(self.temp_db.grid)
        customer_obj = BaseCustomer(n_index, self.temp_db, self.depot_parameter, coordinates)
        self.temp_db.add_customer(customer_obj)


    def create_nodes(self):

        n_index = 0
        for i in range(param_interpret(self.num_depots)):
            self.create_depot(n_index)
            n_index += 1

        for i in range(param_interpret(self.num_customers)):
            self.create_customer(n_index)
            n_index += 1

# Depots:
# ----------------------------------------------------------------------------------------------------------------

class BaseDepot:

    def __init__(self, n_index, temp_db, d_param, coordinates):

        self.node_index  = node_index
        self.coordinates = coordinates

        [setattr(self, k, v) for k, v in d_param.items()]

        self.stock = RestrValueObject('stock', n_index, temp_db, self.max_stock, 0, self.init_demand, self.signal_list)


# Customers:
# ----------------------------------------------------------------------------------------------------------------

class BaseCustomer:

    def __init__(self, n_index, temp_db, c_param, coordinates):

        self.node_index  = node_index
        self.coordinates = coordinates

        [setattr(self, k, v) for k, v in c_param.items()]

        self.demand = RestrValueObject('demand', n_index, temp_db, self.max_demand, 0, self.init_value, self.signal_list)


