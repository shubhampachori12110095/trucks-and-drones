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

from common_sim_func import param_interpret, random_coordinates, l_ignore_none
from nodes import CustomerCreator, DepotCreator
from vehicles import VehicleCreator
from temp_database import TempDatabase, lookup_db

# gewinn - kosten oder nur -kosten?



# Vehicle Travel
# ----------------------------------------------------------------------------------------------------------------


def transporter_travel(transporter_obj, loaded_v_obj_list, coordinates):
    new_coordinates = transporter_obj.travel_to(coordinates)
    [v.set_coordinates(new_coordinates) for v in loaded_v_obj_list]


def simple_travel(vehicle_obj, coordinates):
    vehicle_obj.travel_to(coordinates)



# Vehicle with vehicle interactions:
# ----------------------------------------------------------------------------------------------------------------


def v_unload_v(transporter_obj,to_unload_obj_list, cargo_amount_list):
    '''
    - only unload when cargo can be also unloaded
    - unload nevertheless (ergänzen)
    '''

    # first check how many UV can be unloaded:
    num_v_to_unload = min(l_ignore_none([transporter_obj.cargo_obj.vehicle_per_step, len(to_unload_obj_list)]))

    unloaded_list = []
    for i in range(num_v_to_unload):

        weight = to_unload_obj_list[i].weight

        cargo_amount = min(
            to_unload_obj_list[i].cargo_obj.cargo_per_step.check_subtract_value(cargo_amount_list[i]),
            to_unload_obj_list[i].cargo_obj.standard_cargo.check_add_value(cargo_amount_list[i]),
            transporter_obj.cargo_obj.cargo_per_step.check_subtract_value(cargo_amount_list[i]+weight),
            transporter_obj.cargo_obj.standard_cargo.check_subtract_value(cargo_amount_list[i]+weight)
            )

        if cargo_amount > 0:
            to_unload_obj_list[i].cargo_obj.cargo_per_step.subtract_value(cargo_amount-weight),
            to_unload_obj_list[i].cargo_obj.standard_cargo.add_value(cargo_amount-weight),
            transporter_obj.cargo_obj.cargo_per_step.subtract_value(cargo_amount),
            transporter_obj.cargo_obj.standard_cargo.subtract_value(cargo_amount)

            unloaded_list.append(to_unload_obj_list[i].name)

        else:
            break

    return unloaded_list


def v_load_v(transporter_obj, to_load_obj, cargo_amount):

    if transporter_obj.cargo_obj.vehicle_per_step == None or transporter_obj.cargo_obj.vehicle_per_step >= 1:

        weight = to_load_obj.weight

        cargo_amount = min(
            to_load_obj.cargo_obj.cargo_per_step.check_subtract_value(cargo_amount),
            to_load_obj.cargo_obj.standard_cargo.check_subtract_value(cargo_amount),
            transporter_obj.cargo_obj.cargo_per_step.check_subtract_value(cargo_amount+weight),
            transporter_obj.cargo_obj.standard_cargo.check_add_value(cargo_amount_list[i]+weight)
            )

        if cargo_amount == to_load_obj_list[i].cargo_obj.standard_cargo.cur_value:
            to_load_obj.cargo_obj.cargo_per_step.subtract_value(cargo_amount-weight),
            to_load_obj.cargo_obj.standard_cargo.subtract_value(cargo_amount-weight),
            transporter_obj.cargo_obj.cargo_per_step.subtract_value(cargo_amount),
            transporter_obj.cargo_obj.standard_cargo.add_value(cargo_amount)
            return True

    return False

'''
def v_load_v(transporter_obj, to_load_obj_list, cargo_amount_list):
    '''
    '''

    num_v_to_load = min(l_ignore_none([transporter_obj.cargo_obj.vehicle_per_step, len(to_load_obj_list)]))

    loaded_list = []
    for i in range(num_v_to_load):

        weight = to_load_obj_list[i].weight

        cargo_amount = min(
            to_load_obj_list[i].cargo_obj.cargo_per_step.check_subtract_value(cargo_amount_list[i]),
            to_load_obj_list[i].cargo_obj.standard_cargo.check_subtract_value(cargo_amount_list[i]),
            transporter_obj.cargo_obj.cargo_per_step.check_subtract_value(cargo_amount_list[i]+weight),
            transporter_obj.cargo_obj.standard_cargo.check_add_value(cargo_amount_list[i]+weight)
            )

        if cargo_amount >= to_load_obj_list[i].cargo_obj.standard_cargo.cur_value:
            to_load_obj_list[i].cargo_obj.cargo_per_step.subtract_value(cargo_amount-weight),
            to_load_obj_list[i].cargo_obj.standard_cargo.subtract_value(cargo_amount-weight),
            transporter_obj.cargo_obj.cargo_per_step.subtract_value(cargo_amount),
            transporter_obj.cargo_obj.standard_cargo.add_value(cargo_amount)

            loaded_list.append(to_load_obj_list[i].name)

    return loaded_list
'''


# Vehicle at Node
# ----------------------------------------------------------------------------------------------------------------


def vehicle_at_customer(vehicle_obj, customer_obj, amount):

    amount = min(
        vehicle_obj.cargo_obj.cargo_per_step.check_subtract_value(amount), 
        vehicle_obj.cargo_obj.standard_cargo.check_subtract_value(amount), 
        customer_obj.demand.cur_value
        )

    vehicle_obj.cargo_obj.standard_cargo.subtract_value(amount)
    vehicle_obj.cargo_obj.cargo_per_step.subtract_value(amount)
    customer_obj.demand.subtract_value(amount)
    return amount


def vehicle_at_depot(vehicle_obj, depot_obj, amount):

    amount = min(
        vehicle_obj.cargo_obj.cargo_per_step.check_subtract_value(amount), 
        vehicle_obj.cargo_obj.standard_cargo.check_add_value(amount), 
        depot_obj.stock.cur_value
        )

    vehicle_obj.cargo_obj.standard_cargo.add_value(amount)
    vehicle_obj.cargo_obj.cargo_per_step.subtract_value(amount)
    depot_obj.stock.subtract_value(amount)
    return amount
 

# Functions for simulation init:
# ----------------------------------------------------------------------------------------------------------------


def create_param_name(all_parameter_list):
    return {
        'MV_cargo_param' : all_parameter_list[0],
        'MV_range_param' : all_parameter_list[1],
        'MV_travel_param': all_parameter_list[2], 
        'UV_cargo_param' : all_parameter_list[3],
        'UV_range_param' : all_parameter_list[4],
        'UV_travel_param': all_parameter_list[5],
        'customer_param' : all_parameter_list[6],
        'depot_param'    : all_parameter_list[7],
        }



# Simulation:
# ----------------------------------------------------------------------------------------------------------------
'''
- ergänze funktionen für den fall das jedes vehicle einzeln bewegt wird

'''
def simulation_parameter(
        grid          = [10,10],
        coord_type    = 'exact',
        locked_travel = False,
        num_MV        = 2,
        num_UV_per_MV = 2
        ):
    return {
        'grid'         : grid,
        'coord_type'   : coord_type,
        'locked_travel': locked_travel,
        'num_MV'       : num_MV,
        'num_UV_per_MV': num_UV_per_MV,
        }


class BaseSimulator:

    def __init__(self, all_parameter_list, sim_param):

        self.grid          = grid
        self.coord_type    = coord_type
        self.locked_travel = locked_travel
        self.num_MV        = num_MV
        self.num_UV_per_MV = num_UV_per_MV

        [setattr(self, k, v) for k, v in create_param_name(all_parameter_list).items()]
        [setattr(self, k, v) for k, v in sim_param.items()]


    def reset_simulation(self):

        self.temp_db = TempDatabase(self.grid)

        self.vehicle_list = VehicleCreator(
            self.temp_db, self.coord_type, self.locked_travel,
            # Manned Vehicles:
            self.MV_cargo_param, self.MV_range_param, self.MV_travel_param,
            # Unmanned Vehicles:
            self.UV_cargo_param, self.UV_range_param, self.UV_travel_param).create_vehicles(param_interpret(self.num_MV ),param_interpret(self.num_UV_per_MV))

        self.customer_list = CustomerCreator(self.temp_db, self.customer_param).create_customer_list()

        self.depot_list = DepotCreator(self.temp_db, self.depot_param).create_depot_list()

        self.temp_db.reset_db()
        
        return self.temp_db


    def move(self, vehicle_i, coordinates):
        
        # Check if UV is moveable:
        if any('vehicle_'+str(vehicle_i) in elem for elem in self.temp_db.free_vehicles):
            vehicle     = self.temp_db.base_groups['vehicles'][vehicle_i]
            transp_list = self.temp_db.v_transporting_v['vehicle_'+str(vehicle_i)]

            # Check if vehicle transports other vehicles
            if any(transp_list):
                transporter_travel(vehicle, lookup_db(self.temp_db.base_groups['vehicles'], transp_list), coordinates)
            else:
                simple_travel(vehicle, coordinates)

            self.temp_db.action_signal['free_to_travel'][i] += 1

        else:
            self.temp_db.action_signal['free_to_travel'][i] -= 1

            

    def unload_vehicles(self, vehicle_i, num_v, cargo_amount_list): 
        
        if any('vehicle_'+str(vehicle_i) in elem for elem in self.temp_db.free_vehicles):
            # Get vehicles:
            vehicle = self.temp_db.base_groups['vehicles'][vehicle_i]
            UV_list = lookup_db(self.temp_db.base_groups['vehicles'], self.temp_db.v_transporting_v['vehicle_'+str(vehicle_i)])

            # try to unload UVs from MV i
            unloaded_list = v_unload_v(vehicle, UV_list, num_v, cargo_amount_list)
            
            # Update Error Signal:
            # - positve value: more actions were correct than incorrect
            # - 0: equal number of good and bad actions
            # - negative value: more actions were incorrect
            self.temp_db.action_signal['unloading_v'][i] += (num_v - ((num_v-len(unloaded_list)) * 2))
            
            # Update Database:
            for elem in unloaded_list:
                self.temp_db.v_transporting_v['vehicle_'+str(vehicle_i)].remove(elem)
                self.temp_db.free_vehicles.append(elem)

            self.temp_db.action_signal['free_to_unload_v'][i] += 1

        else:
            self.temp_db.action_signal['free_to_unload_v'][i] -= 1



    def load_vehicle(self, vehicle_i, vehicle_j, cargo_amount):

        if any('vehicle_'+str(vehicle_i) in elem for elem in self.temp_db.free_vehicles):

            v_to_load = self.temp_db.base_groups['vehicles'][vehicle_j]
            
            if v_to_load.loadable:
                loaded = v_load_v(self.temp_db.base_groups['vehicles'][vehicle_i], v_to_load, cargo_amount)

            if loaded and v_to_load.loadable:
                self.temp_db.v_transporting_v['vehicle_'+str(vehicle_i)].append('vehicle_'+str(vehicle_j))
                self.temp_db.action_signal['free_to_be_loaded_v'][j] += 1
            else:
                self.temp_db.action_signal['free_to_be_loaded_v'][j] -= 1

            self.temp_db.action_signal['free_to_load_v'][i] += 1

        else:
            self.temp_db.action_signal['free_to_load_v'][i] -= 1


    def unload_cargo(self, vehicle_i, customer_j, amount):

        if any('vehicle_'+str(vehicle_i) in elem for elem in self.temp_db.free_vehicles):

            real_amount = vehicle_at_customer(self.temp_db.base_groups['vehicles'][vehicle_i], self.temp_db.base_groups['customers'][customer_j], amount)
            self.temp_db.action_signal['free_to_unload_cargo'][i] += 1

        else:
            self.temp_db.action_signal['free_to_unload_cargo'][i] -= 1


    def load_cargo(self, vehicle_i, depot_j, amount):
        
        if any('vehicle_'+str(vehicle_i) in elem for elem in self.temp_db.free_vehicles):

            real_amount = vehicle_at_depot(self.temp_db.base_groups['vehicles'][vehicle_i], self.temp_db.base_groups['depots'][depot_j], amount)
            self.temp_db.action_signal['free_to_load_cargo'][i] += 1

        else:
            self.temp_db.action_signal['free_to_load_cargo'][i]update_signal -= 1


    #def finish_step(self):

    def finish_episode(self):
        # force return to depots for tsp










