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

def customer_parameter(customer_type      = 'static',
                       num_customers      = [5,10],
                       init_demand        = [1,1],
                       first_demand_step  = [0,0],
                       demand_after_steps = None,
                       demand_add         = 1,
                       signal_list        = [1,1-1],
                       ):
    return {
            'customer_type': customer_type,
            'num_custermers': num_custermers,
            'init_demand': demand,
            'first_demand_step': first_demand_step,
            'demand_after_steps': demand_after_steps,
            'demand_add': demand_add,
            'signal_list': signal_list,
            }

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



def MV_unload_UV(MV_vehicle_obj,UV_vehicle_obj_list):

    UV_max_amount = vehicle_obj.cargo_obj.

    vehicle_obj.cargo_obj.unload_UV_cargo()

def MV_load_UV(MV_vehicle_obj, UV_vehicle_obj):



def MV_travel(MV_vehicle_obj, UV_vehicle_obj_list, destination):
    new_coordinates = MV_vehicle_obj.travel_to(destination)
    [UV_vehicle_obj.set_coordinates(new_coordinates) for UV_vehicle_obj in UV_vehicle_obj_list]

def UV_travel(UV_vehicle_obj, destination):
    MV_vehicle_obj.travel_to(destination)


class TraceAndLog:

    def __init__(self):
        
        self.trace_restr_dict = {}
        self.trace_vehicle_dict = {}

    def add_restriction(tracer):
        self.trace_restr_dict[tracer.value_name] = tracer

    def add_vehicle(tracer):
        self.trace_vehicle_dict[tracer.vehicle_name] = tracer


class TrainingLogger(TraceAndLog):
    def __init__(self):
        super().__init__()

class TestingLogger(TraceAndLog):
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

