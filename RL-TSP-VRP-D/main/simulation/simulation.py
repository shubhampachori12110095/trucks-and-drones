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

from main.simulation.common_sim_func import param_interpret, l_ignore_none
from main.simulation.nodes import NodeCreator
from main.simulation.vehicles import VehicleCreator
from main.simulation.temp_database import TempDatabase, lookup_db

# gewinn - kosten oder nur -kosten?


# Vehicle with vehicle interactions:
# ----------------------------------------------------------------------------------------------------------------


def v_unload_v(transporter_obj, to_unload_obj_list):
    '''
    - only unload when cargo can be also unloaded
    - unload nevertheless (ergänzen)
    '''

    # first check how many UV can be unloaded:
    num_v_to_unload = min(l_ignore_none([transporter_obj.cargo_obj.vehicle_per_step, len(to_unload_obj_list)]))

    unloaded_list = []
    for i in range(num_v_to_unload):

        weight = to_unload_obj_list[i].weight
        cargo_amount = to_load_obj.to_unload_obj_list[i].standard_cargo.max_cargo

        cargo_amount = min(
            to_unload_obj_list[i].cargo_obj.cargo_per_step.check_subtract_value(cargo_amount),
            to_unload_obj_list[i].cargo_obj.standard_cargo.check_add_value(cargo_amount),
            transporter_obj.cargo_obj.cargo_per_step.check_subtract_value(cargo_amount+weight),
            transporter_obj.cargo_obj.standard_cargo.check_subtract_value(cargo_amount+weight)
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


def v_load_v(transporter_obj, to_load_obj):

    if transporter_obj.cargo_obj.vehicle_per_step == None or transporter_obj.cargo_obj.vehicle_per_step >= 1:

        weight = to_load_obj.weight
        cargo_amount = to_load_obj.cargo_obj.standard_cargo.cur_value

        cargo_amount = min(
            to_load_obj.cargo_obj.cargo_per_step.check_subtract_value(cargo_amount),
            to_load_obj.cargo_obj.standard_cargo.check_subtract_value(cargo_amount),
            transporter_obj.cargo_obj.cargo_per_step.check_subtract_value(cargo_amount+weight),
            transporter_obj.cargo_obj.standard_cargo.check_add_value(cargo_amount_list[i]+weight)
            )

        if cargo_amount == to_load_obj.cargo_obj.standard_cargo.cur_value:
            to_load_obj.cargo_obj.cargo_per_step.subtract_value(cargo_amount-weight),
            to_load_obj.cargo_obj.standard_cargo.subtract_value(cargo_amount-weight),
            transporter_obj.cargo_obj.cargo_per_step.subtract_value(cargo_amount),
            transporter_obj.cargo_obj.standard_cargo.add_value(cargo_amount)
            return True

    return False



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

        [setattr(self, k, v) for k, v in sim_param.items()]

        self.temp_db = TempDatabase(self.grid)

        all_param_dict = create_param_name(all_parameter_list)


        self.vehicle_creator = VehicleCreator(
            self.temp_db,
            # Manned Vehicles:
            all_param_dict['MV_cargo_param'], all_param_dict['MV_range_param'], all_param_dict['MV_travel_param'],
            # Unmanned Vehicles:
            all_param_dict['UV_cargo_param'], all_param_dict['UV_range_param'], all_param_dict['UV_travel_param']
        )

        self.node_creator = NodeCreator(self.temp_db, all_param_dict['customer_param'], all_param_dict['depot_param'])


    def reset_simulation(self):

        self.temp_db.init_db()

        self.vehicle_creator.create_vehicles(param_interpret(self.num_MV ),param_interpret(self.num_UV_per_MV))
        self.node_creator.create_nodes()

        self.temp_db.reset_db()


    def set_destination(self, vehicle_i, coordinates):

        if coordinates is None:
            coordinates = self.temp_db.nearest_neighbour(vehicle_i, ['c_coord','v_coord'])
        
        self.temp_db.base_groups['vehicles'][vehicle_i].update_destination(coordinates)


    def unload_vehicles(self, vehicle_i, num_v):

        if num_v is None:
            num_v = self.temp_db.v_transporting_v['vehicle_'+str(vehicle_i)]

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
            self.temp_db.status_dict['v_free'][elem.v_index] = 1


    def load_vehicle(self, vehicle_i, vehicle_j):

        if vehicle_j is None:
            vehicle_j = self.temp_db.nearest_neighbour(vehicle_i, 'v_coord')

        if self.temp_db.same_coord(vehicle_j, vehicle_i, 'v_coord'):

            v_to_load = self.temp_db.base_groups['vehicles'][vehicle_j]
            
            if v_to_load.loadable:
                loaded = v_load_v(self.temp_db.base_groups['vehicles'][vehicle_i], v_to_load, cargo_amount)

            if loaded and v_to_load.loadable:
                self.temp_db.status_dict['v_free'][vehicle_j] = 0
                self.temp_db.v_transporting_v['vehicle_'+str(vehicle_i)].append('vehicle_'+str(vehicle_j))
                self.temp_db.action_signal['free_to_be_loaded_v'][j] += 1
            else:
                self.temp_db.action_signal['free_to_be_loaded_v'][j] -= 1


    def unload_cargo(self, vehicle_i, customer_j, amount):

        if customer_j is None:
            customer_j = self.temp_db.nearest_neighbour(vehicle_i,'c_coord')

        if self.temp_db.same_coord(vehicle_j, customer_j, 'c_coord'):
            real_amount = vehicle_at_customer(self.temp_db.base_groups['vehicles'][vehicle_i], self.temp_db.base_groups['customers'][customer_j], amount)


    def load_cargo(self, vehicle_i, depot_j, amount):

        if depot_j is None:
            depot_j = self.temp_db.nearest_neighbour(vehicle_i,'d_coord')
    
        if self.temp_db.same_coord(vehicle_j, depot_j, 'd_coord'):
            real_amount = vehicle_at_depot(self.temp_db.base_groups['vehicles'][vehicle_i], self.temp_db.base_groups['depots'][depot_j], amount)


    def recharge_range(self, vehicle_i):

        #charge_station = 

        if any(self.temp_db.status_dict['v_coord'][vehicle_i] for elem in recharge_coord):
            recharge_v(self.temp_db.base_groups[vehicle_i])


    def finish_step(self):

        next_step_time = min(self.temp_db.times_till_destination)
        self.temp_db.cur_v_index = argmin(self.temp_db.times_till_destination)

        if next_step_time != 0:
            [vehicle.travel_period]

    #def finish_episode(self):
        # force return to depots for tsp






