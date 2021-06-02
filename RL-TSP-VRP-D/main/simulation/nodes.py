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
from common_sim_func import param_interpret, random_coordinates


# Depot:
# ----------------------------------------------------------------------------------------------------------------

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

class CustomerCreator:

    def __init__(self, temp_db, customer_parameter):
        self.temp_db            = temp_db
        self.customer_parameter = customer_parameter
        self.num_customers      = customer_parameter['num_customers']
        del self.customer_parameter['num_customers']

    def create_customer(self, name, i):
        c_param = {}
        for key in self.customer_parameter.keys():
            c_param[key] = param_interpret(self.customer_parameter[key])
            coordinates  = random_coordinates(self.temp_db.grid)
        customer_obj = BaseCustomer(name, i, self.temp_db, c_param, coordinates)
        self.temp_db.add_node(customer_obj)
        return customer_obj
    
    def create_customer_list(self):
        customer_list = []
        [customer_list.append(create_customer('customer_'+str(i),i)) for i in range(self.num_customers)]
        return customer_list

class BaseCustomer:

    def __init__(self, name, node_index, temp_db, c_param, coordinates):

        self.node_index  = node_index
        self.coordinates = coordinates

        [setattr(self, k, v) for k, v in c_param.items()]

        self.demand = RestrValueObject(self.max_demand, 0, self.init_value, self.signal_list)
        temp_db.add_restriction(self.demand, name, base_group='demand')

        if self.demand_after_steps is not None:


# Depot:
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

class DepotCreator:

    def __init__(self, temp_db, depot_parameter):
        self.temp_db         = temp_db
        self.depot_parameter = depot_parameter
        self.num_depots      = param_interpret(depot_parameter['num_depots'])
        del self.depot_parameter['num_depots']

    def create_depot(self, name, i):
        d_param = {}
        for key in self.depot_parameter.keys():
            d_param[key] = param_interpret(self.depot_parameter[key])
            coordinates  = random_coordinates(self.temp_db.grid)
        depot_obj = BaseDepot(name, i, self.temp_db, d_param, coordinates)
        self.temp_db.add_node(depot_obj)
        return depot_obj

    def create_depot_list(self):
        depot_list = []
        [depot_list.append(create_depot('depot_'+str(i), i)) for i in range(self.num_depots)]
        return depot_list

class BaseDepot:

    def __init__(self, name, node_index, temp_db, d_param, coordinates):

        self.node_index  = node_index
        self.coordinates = coordinates

        [setattr(self, k, v) for k, v in d_param.items()]

        self.stock = RestrValueObject(self.max_stock, 0, self.init_demand, self.signal_list)
        temp_db.add_restriction(self.stock, name, base_group='stock')















'''
class BaseCustomerDemand:

    def __init__(self, customer_name, temp_db, demand):

        self.customer_name = customer_name
        self.temp_db        = temp_db
        self.demand        = demand

    def step(self, day_step):
        self.temp_db.update_node(self.customer_name,self.demand.value)


class RechargingCustomerDemand():

    def __init__(self, customer_name, temp_db, demand, demand_after_steps, demand_add):

        self.customer_name = customer_name
        self.temp_db        = temp_db
        self.demand        = demand

        self.demand_after_steps = demand_after_steps
        self.demand_add         = demand_add
        self.step_count         = 0

    def step(self, day_step, count=True):
        if count:
            if self.step_count == self.demand_after_steps:
                self.demand.add_value(self.demand_add)
                self.step_count = 0
            else:
                self.step_count += 1
        self.temp_db.update_node(self.customer_name,self.demand.value)


class BaseCustomerClass:

    def __init__(self, customer_name, temp_db, customer_param):

        self.customer_name = customer_name
        self.temp_db = temp_db

        self.demand = RestrValueObject(customer_name, temp_db, max_restr=self.init_demand, min_restr=0, init_value=self.init_demand, signal_list=self.signal_list)

        if self.demand_after_steps == None:
            self.demand_base_obj = BaseCustomerDemand(customer_name, temp_db, self.demand)
        else:
            self.demand_base_obj = RechargingCustomerDemand(customer_name, temp_db, self.demand, self.demand_after_steps, self.demand_add)

    def unload_vehicle(self,amount):
        return self.demand.subtract_value(amount)

    def step(self, day_step):
        self.demand_base_obj.step(day_step)
        


class DynamicCustomerClass(BaseCustomerClass):

    def __init__(self, customer_name, temp_db, customer_param):
        super().__init__(customer_name, temp_db, customer_param)

        self.step_counter = 0
    
    def step(self, day_step):

        if day_step < self.first_demand_step:
            self.demand.zero_demand()
            self.demand_base_obj.step(day_step, False)
        
        elif day_step == self.first_demand_step:
            self.demand.set_to_max()
            self.demand_base_obj.step(day_step, False)
        
        else:
            self.demand_base_obj.step(day_step)

'''