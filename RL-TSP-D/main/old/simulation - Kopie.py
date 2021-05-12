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

from common_sim_func import param_interpret
from nodes import CustomerCreator, DepotCreator
from vehicles import VehicleCreator
from temp_database import TempDatabase, lookup_db



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
    [vehicle_obj.cargo_obj.cargo_per_step.reset() for vehicle_obj in vehicle_obj_list]
    [vehicle_obj.cargo_obj.cargo_UV_per_step.reset() for vehicle_obj in vehicle_obj_list]


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
'''
- ergänze funktionen für den fall das jedes vehicle einzeln bewegt wird

'''
class BaseSimulator:

    def __init__(self, all_parameter_list, grid=[10,10], coord_type='exact', locked_travel=False, num_MV=2, num_UV_per_MV=2):

        self.grid          = grid
        self.coord_type    = coord_type
        self.locked_travel = locked_travel
        self.num_MV        = num_MV
        self.num_UV_per_MV = num_UV_per_MV

        [setattr(self, k, v) for k, v in create_param_name(all_parameter_list).items()]


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

        return self.temp_db


    def init_step(self):


    def action_move_vehicles(self, coord_list_MV, coord_list_UV):
        '''
        action_list_MV (list): List of tuples representing coordinates (or list of int representing nodes) for MV
        action_list_UV (list): List of tuples representing coordinates (or list of int representing nodes) for UV
        '''

        # Get indices of coordinates for vehicles that should move,
        # None element in coordinates list represents not moving:
        MV_coord_index_list = [i for i, elem in enumerate(coord_list_MV) if elem != None]
        UV_coord_index_list = [i for i, elem in enumerate(coord_list_UV) if elem != None]

        # Get vehicles:
        MV_list = self.temp_db.base_groups_vehicles['MV']
        UV_list = self.temp_db.base_groups_vehicles['UV']

        # Get indices of UV outside:
        UV_outside_index_list = [name[-1] for name in self.temp_db.UV_outside]
        # Determine which UV can move (Uv must be outside and coordinates not None):
        UVs_moving = list(set(UV_outside_index_list).intersection(UV_coord_index_list))

        # Move vehicles:
        [MV_travel(MV_list[i], lookup_db(UV_list,self.temp_db.MV_transporting_UV['MV_'+str(i)]), coord_list_MV[i]) for i in MV_coord_index_list]
        [UV_travel(UV_list[i], coord_list_UV[i]) for i in UVs_moving]

        # return indicator of how successfull the move action for UV was 
        #(1 means UV could move, -1 could not move):
        # Be careful in penalizing this since this could result in learning to never move UVs.
        UVs_not_moving = list(set(UV_outside_index_list).difference(UV_coord_index_list))
        [self.temp_db.action_signal['UV_'+str(i)+'_travel'].append(1) for i in UVs_moving]
        [self.temp_db.action_signal['UV_'+str(i)+'_travel'].append(-1) for i in UVs_not_moving]


    def single_action_move_MV(self, vehicle_index, coordinates):
        # Get vehicles:
        MV      = self.temp_db.base_groups_vehicles['MV'][vehicle_index]
        UV_list = lookup_db(self.temp_db.base_groups_vehicles['UV'], self.temp_db.MV_transporting_UV['MV_'+str(vehicle_index)])
        # Move MV:
        MV_travel(MV, UV_list, coordinates)


    def single_action_move_UV(self, vehicle_index, coordinates):
        # Check if UV is moveable:
        if any('UV_'+str(vehicle_index) in elem for elem in self.temp_db.UV_outside):
            UV_travel(self.temp_db.base_groups_vehicles['UV'][vehicle_index], coordinates)
            # indicator that UV movement was possible:
            self.temp_db.action_signal['UV_'+str(vehicle_index)+'_travel'].append(1)
        else:
            # indicator that UV movement was NOT possible:
            self.temp_db.action_signal['UV_'+str(vehicle_index)+'_travel'].append(-1)


    def action_unload_UVs(self, MV_index_and_numUVs_list):
        [self.single_action_unload_UVs(index_and_numUVs) for index_and_numUVs in MV_index_and_numUVs_list]
            

    def single_action_unload_UVs(self, index_and_numUVs):

        # Index of MV:
        i = index_and_numUVs[0]

        # Number of UVs to unload:
        numUVs = index_and_numUVs[1]

        # Get vehicles:
        MV      = self.temp_db.base_groups_vehicles['MV'][i]
        UV_list = lookup_db(self.temp_db.base_groups_vehicles['UV'], self.temp_db.MV_transporting_UV['MV'+str(i)])

        # try to unload UVs from MV i
        unloaded_list = MV_unload_UV(MV, UV_list, numUVs)
        
        # Update Error Signal:
        # - positve value: more actions were correct than incorrect
        # - 0: equal number of good and bad actions
        # - negative value: more actions were incorrect
        self.temp_db.action_signal['MV_'+str(i)+'_unloading_UV'].append(numUVs - ((numUVs-len(unloaded_list)) * 2))
        
        # Update Database:
        for elem in unloaded_list:
            self.temp_db.MV_transporting_UV['MV'+str(i)].remove(elem)
            self.temp_db.UV_outside.append(elem)


    def action_load_UVs(self, MV_UV_index_pair_list):
        [self.single_action_load_UVs(MV_UV_index_pair) for MV_UV_index_pair in MV_UV_index_pair_list]
            
        
    def single_action_load_UVs(self, MV_UV_index_pair):
        MV_i = MV_UV_index_pair[0]
        UV_i = MV_UV_index_pair[1]
        MV_load_UV(self.temp_db.base_groups_vehicles['MV'][MV_i], self.temp_db.base_groups_vehicles['UV'][UV_i])


    def action_unload_cargo(self, vehicle_customer_index_pair_list, vehicle_type):
        [self.single_action_unload_cargo(vehicle_customer_index_pair) for vehicle_customer_index_pair in vehicle_customer_index_pair_list]


    def single_action_unload_cargo(self, vehicle_customer_index_pair, vehicle_type):
        V_i = vehicle_customer_index_pair[0]
        C_i = vehicle_customer_index_pair[1]
        MV_load_UV(self.temp_db.base_groups_vehicles[vehicle_type][V_i], self.temp_db.base_groups_vehicles['UV'][C_i])


    def action_load_cargo(self, vehicle_depot_index_pair_list, vehicle_type):
        [self.single_action_load_cargo(vehicle_depot_index_pair) for vehicle_depot_index_pair in vehicle_depot_index_pair_list]


    def single_action_load_cargo(self, vehicle_depot_index_pair, vehicle_type):
        V_i = vehicle_depot_index_pair[0]
        D_i = vehicle_depot_index_pair[1]
        MV_load_UV(self.temp_db.base_groups_vehicles[vehicle_type][V_i], self.temp_db.base_groups_vehicles['UV'][D_i])


    def finish_step(self):













    def action_move(self, vehicle_index, coordinates):
        '''
        coord_index_list = [i for i, elem in enumerate(coord_list) if elem != None]
        [self.single_action_move(self.temp_db.v_index_list[i],coord_index_list[i]) for i in coord_list]
        '''
        # Check if UV is moveable:
        if any('vehicle_'+str(vehicle_index) in elem for elem in self.temp_db.moveable_vehicles):
            vehicle     = self.temp_db.base_groups['vehicles'][vehicle_index]
            transp_list = self.temp_db.v_transporting_v['vehicle_'+str(vehicle_index)]

            if any(transp_list):
                MV_travel(vehicle, lookup_db(self.temp_db.base_groups['vehicles'], self.temp_db.moveable_vehicles['vehicles_'+str(vehicle_index)]), coordinates)
            else:
                UV_travel(vehicle, coordinates)

            self.temp_db.action_signal['vehicles_'+str(vehicle_index)+'_travel'].append(1)

        else:
            self.temp_db.action_signal['vehicles_'+str(vehicle_index)+'_travel'].append(-1)

            

    def action_unload_vehicles(self, index, num_v):
        '''
        [self.single_action_unload_UVs(index_and_numUVs) for index_and_numUVs in MV_index_and_numUVs_list]
        '''
        # Get vehicles:
        vehicle = self.temp_db.base_groups['vehicles'][index]
        UV_list = lookup_db(self.temp_db.base_groups['vehicles'], self.temp_db.v_transporting_v['vehicle_'+str(index)])

        # try to unload UVs from MV i
        unloaded_list = MV_unload_UV(MV, UV_list, num_v)
        
        # Update Error Signal:
        # - positve value: more actions were correct than incorrect
        # - 0: equal number of good and bad actions
        # - negative value: more actions were incorrect
        self.temp_db.action_signal['vehicle_'+str(i)+'_unloading_v'].append(num_v - ((num_v-len(unloaded_list)) * 2))
        
        # Update Database:
        for elem in unloaded_list:
            self.temp_db.v_transporting_v['vehicles'+str(i)].remove(elem)
            self.temp_db.moveable_vehicles.append(elem)


    def action_load_UVs(self, MV_UV_index_pair_list):
        [self.single_action_load_UVs(MV_UV_index_pair) for MV_UV_index_pair in MV_UV_index_pair_list]
            
        
    def single_action_load_UVs(self, MV_UV_index_pair):
        MV_i = MV_UV_index_pair[0]
        UV_i = MV_UV_index_pair[1]
        MV_load_UV(self.temp_db.base_groups_vehicles['MV'][MV_i], self.temp_db.base_groups_vehicles['UV'][UV_i])


    def action_unload_cargo(self, vehicle_customer_index_pair_list, vehicle_type):
        [self.single_action_unload_cargo(vehicle_customer_index_pair) for vehicle_customer_index_pair in vehicle_customer_index_pair_list]


    def single_action_unload_cargo(self, vehicle_customer_index_pair, vehicle_type):
        V_i = vehicle_customer_index_pair[0]
        C_i = vehicle_customer_index_pair[1]
        MV_load_UV(self.temp_db.base_groups_vehicles[vehicle_type][V_i], self.temp_db.base_groups_vehicles['UV'][C_i])


    def action_load_cargo(self, vehicle_depot_index_pair_list, vehicle_type):
        [self.single_action_load_cargo(vehicle_depot_index_pair) for vehicle_depot_index_pair in vehicle_depot_index_pair_list]


    def single_action_load_cargo(self, vehicle_depot_index_pair, vehicle_type):
        V_i = vehicle_depot_index_pair[0]
        D_i = vehicle_depot_index_pair[1]
        MV_load_UV(self.temp_db.base_groups_vehicles[vehicle_type][V_i], self.temp_db.base_groups_vehicles['UV'][D_i])









  
# Simulation:
# ----------------------------------------------------------------------------------------------------------------

class AutomateActions()