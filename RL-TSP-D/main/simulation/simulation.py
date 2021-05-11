'''
- euclidean route
- non-euclidean route

Abbreviations:

MV: Manned Vehicle
UV: Unmanned Vehicle
SoC: State of Charge


unterscheidung ob dinge gleichzeitig oder nacheinander passieren können:
- travel and unload/load
- unload UV and unload cargo for UV


'''
import numpy as np

from nodes import CustomerCreator, DepotCreator
from vehicles import VehicleCreator
from temp_database import TempDatabase



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


# Functions for init:
# ----------------------------------------------------------------------------------------------------------------

def create_param_name(all_parameter_list):
    return {
        'MV_cargo_param': all_parameter_list[0],
        'MV_range_param': all_parameter_list[1],
        'MV_travel_param': all_parameter_list[2], 
        'UV_cargo_param': all_parameter_list[3],
        'UV_range_param': all_parameter_list[4],
        'UV_travel_param': all_parameter_list[5],
        'customer_param': all_parameter_list[6],
        'depot_param': all_parameter_list[7],
        }

# Simulation:
# ----------------------------------------------------------------------------------------------------------------

class BaseSimulator:

    def __init__(self, all_parameter_list, grid=[10,10], coord_type='exact', locked_travel=False):

        self.grid          = grid
        self.coord_type    = coord_type
        self.locked_travel = locked_travel

        [setattr(self, k, v) for k, v in create_param_name(all_parameter_list).items()]


    def reset_simulation(self):

        self.temp_db = TempDatabase(self.grid)

        self.vehicle_list = VehicleCreator(
            self.temp_db, self.coord_type, self.locked_travel,
            # Manned Vehicles:
            self.MV_cargo_param, self.MV_range_param, self.MV_travel_param,
            # Unmanned Vehicles:
            self.UV_cargo_param, self.UV_range_param, self.UV_travel_param)

        self.customer_list = CustomerCreator(self.temp_db, self.customer_param)

        self.depot_list = DepotCreator(self.temp_db, self.depot_param)


    def init_step(self):


    def action_move_vehicles(self, action_list):


    def action_unload_UVs(self, action_list):


    def action_load_UVs(self, action_list):


    def action_unload_vehicles(self, action_list):


    def action_load_vehicles(self, action_list):


    def finish_step(self):

  
