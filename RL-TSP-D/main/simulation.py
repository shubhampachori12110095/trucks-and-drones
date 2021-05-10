import numpy as np

'''
- euclidean route
- non-euclidean route

Abbreviations:

MV: Manned Vehicle
UV: Unmanned Vehicle
SoC: State of Charge
'''

##################################
######## Customer Classes ########
##################################
class BaseCustomerDemand:

    def __init__(self, customer_name, logger, demand):

        self.customer_name = customer_name
        self.logger        = logger
        self.demand        = demand

    def step(self, day_step):
        self.logger.update_node(self.customer_name,self.demand.value)


class RechargingCustomerDemand():

    def __init__(self, customer_name, logger, demand, demand_after_steps, demand_add):

        self.customer_name = customer_name
        self.logger        = logger
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
        self.logger.update_node(self.customer_name,self.demand.value)


class BaseCustomerClass:

    def __init__(self, customer_name, logger, customer_param):

        self.customer_name = customer_name
        self.logger = logger

        self.demand = RestrValueObject(customer_name, logger, max_restr=self.init_demand, min_restr=0, init_value=self.init_demand, signal_list=self.signal_list)

        if self.demand_after_steps == None:
            self.demand_base_obj = BaseCustomerDemand(customer_name, logger, self.demand)
        else:
            self.demand_base_obj = RechargingCustomerDemand(customer_name, logger, self.demand, self.demand_after_steps, self.demand_add)

    def unload_vehicle(self,amount):
        return self.demand.subtract_value(amount)

    def step(self, day_step):
        self.demand_base_obj.step(day_step)
        


class DynamicCustomerClass(BaseCustomerClass):

    def __init__(self, customer_name, logger, customer_param):
        super().__init__(customer_name, logger, customer_param)

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

def param_interpret(variable):
    if isinstance(variable, (list, tuple, np.ndarray)):
        return np.random.randint(variable[0],variable[1]+1)
    else:
        return variable
# Customer Module:
# ----------------------------------------------------------------------------------------------------------------

def customer_parameter(
        customer_type      = 'static',
        num_customers      = [5,10],
        init_demand        = [1,1],
        first_demand_step  = [0,0],
        demand_after_steps = None,
        demand_add         = 1,
        signal_list        = [1,1-1],
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

    def __init__(self, logger, customer_parameter):
        self.logger             =logger
        self.customer_parameter = customer_parameter
        self.num_customers      = customer_parameter['num_customers']
        del self.customer_parameter['num_customers']

        #[setattr(self, k, v) for k, v in customer_parameter.items()]

    def create_customer(self, name):
        c_param = {}
        for key in self.customer_parameter.keys():
            c_param[key] = param_interpret(self.customer_parameter[key])
        return BaseCustomer(name, self.logger, c_param)

    def create_customer_list(self):
        customer_list = []
        [customer_list.append(create_customer('customer_'+str(i))) for i in range(self.num_customers)]
        return customer_list


# Depot Module:
# ----------------------------------------------------------------------------------------------------------------


# Vehicle at Node
# ----------------------------------------------------------------------------------------------------------------

def vehicle_at_customer(vehicle_obj, customer_obj, amount):

    amount = min(
        vehicle_obj.cargo_obj.cargo_per_step, 
        vehicle_obj.cargo_obj.standard_cargo.check_subtract_value(amount), 
        customer_obj.demand.tracer.current_value
        )

    vehicle_obj.cargo_obj.standard_cargo.subtract_value(amount)
    customer_obj.demand.subtract_value(amount)

def vehicle_at_depot(vehicle_obj, depot_obj, amount):

    amount = min(
        vehicle_obj.cargo_obj.cargo_per_step, 
        vehicle_obj.cargo_obj.standard_cargo.check_add_value(amount), 
        depot_obj.stock.tracer.current_value
        )

    vehicle_obj.cargo_obj.standard_cargo.add_value(amount)
    depot_obj.stock.subtract_value(amount)

# Vehicle Travel
# ----------------------------------------------------------------------------------------------------------------

def MV_travel(MV_obj, UV_obj_list, destination):
    new_coordinates = MV_obj.travel_to(destination)
    [UV_obj.set_coordinates(new_coordinates) for UV_obj in UV_obj_list]

def UV_travel(UV_obj, destination):
    MV_obj.travel_to(destination)

# Run each Step
# ----------------------------------------------------------------------------------------------------------------

def reset_rates(vehicle_obj_list):
    [vehicle_obj.cargo_obj.cargo_per_step.reset(),vehicle_obj.cargo_obj.cargo_UV_per_step.reset() for vehicle_obj in vehicle_obj_list]


# MV and UV interactions:
# ----------------------------------------------------------------------------------------------------------------


def MV_unload_UV(MV_obj,UV_obj_list):
    '''
    - only unload when cargo can be also unloaded
    - unload nevertheless
    '''

    # first check how many UV can be unloaded:
    num_UV_to_unload = min(MV_obj.cargo_obj.cargo_UV_per_step, len(UV_obj_list))
    init

    for i in range(num_UV_to_unload):

        cargo_amount = min(
            UV_obj_list[i].cargo_obj.cargo_per_step,
            )

    # second check unload cargo for UV

    UV_num_to_unload = vehicle_obj
    UV_weight        = vehicle_obj.cargo_obj.

    vehicle_obj.cargo_obj.unload_UV_cargo()

def MV_load_UV(MV_obj, UV_obj):


# Database Management with Tracer and Logger:
# ----------------------------------------------------------------------------------------------------------------

class GlobalTracer:

    def __init__(self):

        self.trace_restr_dict   = {}
        self.trace_vehicle_dict = {}

        self.groups_dict = {}

        self.groups_dict['vehicle'] = []
        self.groups_dict['MV']      = []
        self.groups_dict['UV']      = []

        self.groups_dict['node']             = []
        self.groups_dict['customer']         = []
        self.groups_dict['static_customer']  = []
        self.groups_dict['dynamic_customer'] = []
        self.groups_dict['depot']            = []


        self.groups_dict['range'] = []
        self.groups_dict['SoC']   = []

        self.groups_dict['cargo']    = []
        self.groups_dict['UV_cargo'] = []
        self.groups_dict['MV_cargo'] = []

        self.groups_dict['stock']  = []
        self.groups_dict['demand'] = []

        self.groups_dict['coordinates']          = []
        self.groups_dict['vehicle_coordinates']  = []
        self.groups_dict['MV_coordinates']       = []
        self.groups_dict['UV_coordinates']       = []
        self.groups_dict['node_coordinates']     = []
        self.groups_dict['depot_coordinates']    = []
        self.groups_dict['customer_coordinates'] = []


    def add_restriction(tracer):
        self.trace_restr_dict[tracer.value_name] = tracer

    def add_vehicle(tracer):
        self.trace_vehicle_dict[tracer.vehicle_name] = tracer
        

class TrainingLogger(BaseLogger):
    def __init__(self):
        super().__init__()


class TestingLogger(BaseLogger):
    def __init__(self):
        super().__init__()


###############################
######## Depot Classes ########
###############################

def depot_parameter(num_depots       = [1,1],
                    max_stock        = None,
                    resupply_rate    = 1,
                    unlimited_supply = True,
                    init_value       = 0,
                    signal_list      = [1,1-1],
                    ):
    return {
            'max_stock': max_stock,
            'resupply_rate': resupply_rate,
            'unlimited_supply': unlimited_supply,
            'init_value': init_value,
            'signal_list': signal_list,
            }

class BaseDepotClass:

    def __init__(self, depot_name, logger, depot_parameter, customer_nodes=None, coordinates=None):

        if self.unlimited_supply == True:
            self.stock = RestrValueObject(depot_name, logger, max_restr=None, min_restr=None, init_value=self.init_value, signal_list=self.signal_list)
        else:
            self.stock = RestrValueObject(depot_name, logger, max_restr=self.max_stock, min_restr=0, init_value=self.init_value, signal_list=self.signal_list)


    def place_depot():

    def load_vehicle(amount):
        return self.stock.subtract_value(amount)




class StandardGrid:

    def __init__(self, x_size=100, y_size=100, num_MV=1, num_UV_per_MV=1):

        self.x_size = x_size
        self.y_size = y_size

        self.num_customers = num_customers
        self.num_MV        = num_MV
        self.num_UV_per_MV = num_UV_per_MV

    def 

