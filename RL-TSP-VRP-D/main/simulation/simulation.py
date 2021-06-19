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
from main.simulation.temp_database import TempDatabase


def calc_rates(self, should_value, keys, indices):
    rate_vals = [self.temp_db.status_dict[keys[i]][indices[i]] for i in range(len(keys))]
    min_rate = np.min(rate_vals)
    i_of_min = np.argmin(rate_vals)
    return min(should_value, min_rate), self.temp_db.status_dict[keys[i_of_min]][indices[i_of_min]]

# Move Functions:
# ----------------------------------------------------------------------------------------------------------------

def distance_value(self):
    
    self.temp_db.status_dict['v_direc'][self.v_index] = (
        self.temp_db.status_dict['v_dest'][self.v_index] - self.temp_db.status_dict['v_coord'][self.v_index]
    )
    
    return self.calc_distance(self.temp_db.status_dict['v_direc'][self.v_index])

def distance_rate(self, should_value):
    
    real_distance = np.min(
        should_value,
        self.temp_db.status_dict['distance_in_time'][self.v_index]
    )
    rate_constant = self.temp_db.constants_dict['speed'][self.v_index]
    
    return real_distance, rate_constant

def distance_act(self, real_distance, should_distance):
    
    real_distance = self.go(real_distance)
    self.temp_db.status_dict['v_coord'][self.v_index] = (
        self.temp_db.status_dict['v_direc'][self.v_index] * (real_distance/should_distance) + self.temp_db.status_dict['v_coord'][self.v_index]
    )
    
    return real_distance

move_funcs = [distance_value, distance_rate, distance_act]


# Unload Vehicle Functions:
# ----------------------------------------------------------------------------------------------------------------

def unload_vehicle(self):

    to_unload_index = self.temp_db.status_dict['v_to_unload'][self.v_index]
    weight = self.temp_db.status_dict['weight'][to_unload_index]
    
    return np.min(
        self.temp_db.status_dict['cargo'][self.v_index] - weight,
        self.temp_db.status_dict['max_cargo'][to_unload_index],
    )

def unload_vehicle_rate(self, cargo_amount):

    to_unload_index = self.temp_db.status_dict['v_to_unload'][self.v_index]
    weight = self.temp_db.status_dict['weight'][to_unload_index]
    
    real_cargo_amount = np.min(
        cargo_amount,
        self.temp_db.status_dict['cargo_in_time'][self.v_index] - weight,
        self.temp_db.status_dict['cargo_in_time'][to_unload_index],
    )

    if self.temp_db.constants_dict['cargo_rate'][self.v_index] < self.temp_db.constants_dict['cargo_rate'][to_unload_index]:
        rate_constant = self.temp_db.constants_dict['cargo_rate'][self.v_index]
    else:
        rate_constant = self.temp_db.constants_dict['cargo_rate'][to_unload_index]
    
    return real_distance, rate_constant

def unload_vehicle_act(self, real_cargo_amount, cargo_amount):
    
    to_unload_index = self.temp_db.status_dict['v_to_unload'][self.v_index]
    weight = self.temp_db.status_dict['weight'][to_unload_index]
    
    return np.min(
            self.cargo_rate.subtract_value(real_cargo_amount+weight),
            self.cargo.subtract_value(real_cargo_amount+weight),
            self.temp_db.base_groups[to_unload_index].cargo_rate.subtract_value(real_cargo_amount),
            self.temp_db.base_groups[to_unload_index].cargo.add_value(real_cargo_amount),
        )

unload_vehicle_funcs = [unload_vehicle, unload_vehicle_rate, unload_vehicle_act]


# Load Vehicle Functions:
# ----------------------------------------------------------------------------------------------------------------

def load_vehicle(self):

    to_load_index = self.temp_db.status_dict['v_to_load'][self.v_index]
    weight = self.temp_db.status_dict['weight'][to_load_index]
    
    return np.min(
        self.temp_db.status_dict['cargo'][self.v_index] - weight,
        self.temp_db.status_dict['max_cargo'][to_load_index],
    )

def load_vehicle_rate(self, cargo_amount):

    to_load_index = self.temp_db.status_dict['v_to_load'][self.v_index]
    weight = self.temp_db.status_dict['weight'][to_load_index]
    
    real_cargo_amount = np.min(
        cargo_amount,
        self.temp_db.status_dict['cargo_in_time'][self.v_index] - weight,
        self.temp_db.status_dict['cargo_in_time'][to_load_index],
    )

    if self.temp_db.constants_dict['cargo_rate'][self.v_index] < self.temp_db.constants_dict['cargo_rate'][to_load_index]:
        rate_constant = self.temp_db.constants_dict['cargo_rate'][self.v_index]
    else:
        rate_constant = self.temp_db.constants_dict['cargo_rate'][to_load_index]
    
    return real_distance, rate_constant

def load_vehicle_act(self, real_cargo_amount, cargo_amount):
    
    to_load_index = self.temp_db.status_dict['v_to_load'][self.v_index]
    weight = self.temp_db.status_dict['weight'][to_load_index]
    
    return np.min(
            self.cargo_rate.subtract_value(real_cargo_amount+weight),
            self.cargo.add_value(real_cargo_amount+weight),
            self.temp_db.base_groups[to_load_index].cargo_rate.subtract_value(real_cargo_amount),
            self.temp_db.base_groups[to_load_index].cargo.subtract_value(real_cargo_amount),
        )

load_vehicle_funcs = [load_vehicle, load_vehicle_rate, load_vehicle_act]
# Vehicle with vehicle interactions:
# ----------------------------------------------------------------------------------------------------------------


def v_load_v(self, v_i, v_j):

    cargo_i = self.temp_db.restr_dict['cargo'][v_i]
    items_i = self.temp_db.restr_dict['items'][v_i]
    loaded_v_i = self.temp_db.restr_dict['loaded_v'][v_i]
    
    cargo_j = self.temp_db.restr_dict['cargo'][v_j]
    items_j = self.temp_db.restr_dict['items'][v_j]
    weight_j = self.temp_db.constants_dict['weight'][v_j]

    if restr_i.max_restr - restr_i.cur_value() >= restr_j.cur_value() and (loaded_v_i.cur_value() is None or bool(loaded_v_i.check_add_value(1) - loaded_v_i.cur_value())):
        cargo_amount = restr_j.cur_value()
    
        real_cargo_amount = min(
            cargo_i.check_subtract_value(cargo_amount + weight)
            items_i.check_subtract_value(cargo_amount)
            cargo_j.check_add_value(cargo_amount)
            items_j.check_add_value(cargo_amount)
        )

        if real_cargo_amount > 0:
            real_cargo_amount = min(
                cargo_i.subtract_value(real_cargo_amount + weight)
                items_i.subtract_value(real_cargo_amount)
                cargo_j.add_value(real_cargo_amount)
                items_j.add_value(real_cargo_amount)
            )

        if real_cargo_amount == cargo_amount:
            loaded_v_i.add_value(1):
            self.temp_db.v_transporting_v[v_i].append(v_j)
            self.temp_db.actions_list[v_i].pop(0)
            self.temp_db.time_till_fin[v_i] = None
        
        else:
            self.temp_db.time_till_fin[v_i] = (real_cargo_amount / self.temp_db.cur_time_frame) * (cargo_amount - real_cargo_amount)

    else:
        self.temp_db.actions_list[v_i].pop(0)
        self.temp_db.time_till_fin[v_i] = None


def v_unload_v(transporter_obj, to_load_obj):

    if transporter_obj.cargo_obj.cargo_UV_rate == None or transporter_obj.cargo_obj.cargo_UV_rate.cur_value() >= 1:

        weight = to_load_obj.weight
        cargo_amount = to_load_obj.cargo_obj.standard_cargo.cur_value()

        cargo_amount = min(
            to_load_obj.cargo_obj.cargo_rate.check_subtract_value(cargo_amount),
            to_load_obj.cargo_obj.standard_cargo.check_subtract_value(cargo_amount),
            transporter_obj.cargo_obj.cargo_rate.check_subtract_value(cargo_amount+weight),
            transporter_obj.cargo_obj.standard_cargo.check_add_value(cargo_amount+weight)
            )

        if cargo_amount == to_load_obj.cargo_obj.standard_cargo.cur_value():
            to_load_obj.cargo_obj.cargo_rate.subtract_value(cargo_amount-weight),
            to_load_obj.cargo_obj.standard_cargo.subtract_value(cargo_amount-weight),
            transporter_obj.cargo_obj.cargo_rate.subtract_value(cargo_amount),
            transporter_obj.cargo_obj.standard_cargo.add_value(cargo_amount)
            return True

    return False



# Vehicle at Node
# ----------------------------------------------------------------------------------------------------------------


def vehicle_at_customer(vehicle_obj, customer_obj, amount):

    if amount is None:
        amount = min(
            vehicle_obj.cargo_obj.cargo_rate.cur_value(), 
            vehicle_obj.cargo_obj.standard_cargo.cur_value(), 
            customer_obj.demand.cur_value()
        )
    else:
        amount = min(
            vehicle_obj.cargo_obj.cargo_rate.check_subtract_value(amount), 
            vehicle_obj.cargo_obj.standard_cargo.check_subtract_value(amount), 
            customer_obj.demand.cur_value()
            )

    vehicle_obj.cargo_obj.standard_cargo.subtract_value(amount)
    vehicle_obj.cargo_obj.cargo_rate.subtract_value(amount)
    customer_obj.demand.subtract_value(amount)
    return amount


def vehicle_at_depot(vehicle_obj, depot_obj, amount):

    if amount is None:
        amount = min(
            vehicle_obj.cargo_obj.cargo_rate.cur_value(), 
            vehicle_obj.cargo_obj.standard_cargo.max_restr - vehicle_obj.cargo_obj.standard_cargo.cur_value(), 
            depot_obj.stock.cur_value()
        )

    else:
        amount = min(
            vehicle_obj.cargo_obj.cargo_rate.check_subtract_value(amount), 
            vehicle_obj.cargo_obj.standard_cargo.check_add_value(amount), 
            depot_obj.stock.cur_value()
            )

    

    vehicle_obj.cargo_obj.standard_cargo.add_value(amount)
    vehicle_obj.cargo_obj.cargo_rate.subtract_value(amount)
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

        self.free_after_step = []

    def reset_simulation(self):

        self.temp_db.init_db()

        self.node_creator.create_nodes()
        self.vehicle_creator.create_vehicles(param_interpret(self.num_MV ),param_interpret(self.num_UV_per_MV))
        

        self.temp_db.reset_db()


    def set_destination(self, vehicle_i, coordinates):

        if coordinates is None:
            if self.temp_db.base_groups['vehicles'][vehicle_i].cargo_obj.standard_cargo.cur_value() > 0 and 0 in self.temp_db.status_dict['c_waiting']:
                coord_index = self.temp_db.nearest_neighbour(vehicle_i, ['c_coord'], exclude=[['demand',0], ['c_waiting',1]])
                coordinates = self.temp_db.status_dict['c_coord'][coord_index]
                self.temp_db.status_dict['c_waiting'][coord_index] = 1
                self.temp_db.status_dict['v_to_n'][vehicle_i] = 1
                self.temp_db.status_dict['v_to_n_index'][vehicle_i] = coord_index
            else:
                coord_index = self.temp_db.nearest_neighbour(vehicle_i, ['d_coord'], exclude=[['stock',0]])
                coordinates = self.temp_db.status_dict['d_coord'][coord_index]
                if self.temp_db.status_dict['v_to_n'][vehicle_i] == 1:
                    self.temp_db.status_dict['c_waiting'][self.temp_db.status_dict['v_to_n_index'][vehicle_i]] = 0
                self.temp_db.status_dict['v_to_n'][vehicle_i] = 0
                self.temp_db.status_dict['v_to_n_index'][vehicle_i] = coord_index

        self.temp_db.base_groups['vehicles'][vehicle_i].update_destination(coordinates)

        print('new_destination:',coordinates,'for',vehicle_i)


    def unload_vehicles(self, vehicle_i, num_v):

        if num_v is None:
            num_v = len(self.temp_db.v_transporting_v[vehicle_i])

        # Get vehicles:
        vehicle = self.temp_db.base_groups['vehicles'][vehicle_i]
        UV_list = [self.temp_db.base_groups['vehicles'][i] for i in self.temp_db.v_transporting_v[vehicle_i]]

        # try to unload UVs from MV i
        unloaded_list = v_unload_v(vehicle, UV_list)
        
        # Update Error Signal:
        # - positve value: more actions were correct than incorrect
        # - 0: equal number of good and bad actions
        # - negative value: more actions were incorrect
        self.temp_db.action_signal['unloading_v'][vehicle_i] += (num_v - ((num_v-len(unloaded_list)) * 2))
        
        # Update Database:
        for elem in unloaded_list:
            self.temp_db.v_transporting_v[vehicle_i].remove(elem)
            self.free_after_step.append(elem.v_index)
            self.tempd_db.status_dict['time_to_dest'][elem.v_index] = 0

            print('unloaded v',elem.v_index,'from',vehicle_i)
            self.v_did_sth = True


    def load_vehicle(self, vehicle_i, vehicle_j):

        if vehicle_j is None:
            vehicle_j = self.temp_db.nearest_neighbour(vehicle_i, 'v_coord', exclude=[['v_type',1], ['v_free',0]])

        if self.temp_db.same_coord(vehicle_i, vehicle_j, 'v_coord'):

            v_to_load = self.temp_db.base_groups['vehicles'][vehicle_j]
            
            loaded = False
            if v_to_load.v_loadable:
                loaded = v_load_v(self.temp_db.base_groups['vehicles'][vehicle_i], v_to_load)

            if loaded and v_to_load.v_loadable:
                self.temp_db.status_dict['v_free'][vehicle_j] = 0
                self.temp_db.status_dict['time_to_dest'][vehicle_j] = 10000
                self.temp_db.v_transporting_v[vehicle_i].append(vehicle_j)
                self.temp_db.action_signal['free_to_be_loaded_v'][vehicle_j] += 1
                print('loaded v',v_to_load.v_index,'to',vehicle_i)
                self.v_did_sth = True
            else:
                self.temp_db.action_signal['free_to_be_loaded_v'][vehicle_j] -= 1


    def unload_cargo(self, vehicle_i, customer_j, amount):

        if customer_j is None:
            customer_j = self.temp_db.nearest_neighbour(vehicle_i, 'c_coord', exclude=[['demand',0]])

        if self.temp_db.same_coord(vehicle_i, customer_j, 'c_coord'):
            real_amount = vehicle_at_customer(self.temp_db.base_groups['vehicles'][vehicle_i], self.temp_db.base_groups['customers'][customer_j], amount)

            print('unloaded cargo',real_amount,'from',vehicle_i,'to customer',customer_j)
            if real_amount > 0:
                self.v_did_sth = True

    def load_cargo(self, vehicle_i, depot_j, amount):

        if depot_j is None:
            depot_j = self.temp_db.nearest_neighbour(vehicle_i,'d_coord', exclude=[['stock',0]])
    
        if self.temp_db.same_coord(vehicle_i, depot_j, 'd_coord'):
            real_amount = vehicle_at_depot(self.temp_db.base_groups['vehicles'][vehicle_i], self.temp_db.base_groups['depots'][depot_j], amount)

            print('loaded cargo',real_amount,'to',vehicle_i, 'from depot', depot_j)
            if real_amount > 0:
                self.v_did_sth = True

    def recharge_range(self, vehicle_i):

        #charge_station = 

        if any(self.temp_db.status_dict['v_coord'][vehicle_i] for elem in recharge_coord):
            recharge_v(self.temp_db.base_groups[vehicle_i])


    def finish_step(self):

        print('free_after_step', self.free_after_step)
        print('cur_v_index', self.temp_db.cur_v_index)
        print('v_transporting_v', self.temp_db.v_transporting_v)

        if any(self.free_after_step):
            self.temp_db.status_dict['v_free'][self.free_after_step[0]] = 1
            self.temp_db.cur_v_index = self.free_after_step[0]
            self.free_after_step.pop(0)
        
        else:

            print('v did sth', self.v_did_sth)

            next_step_time = min(self.temp_db.status_dict['time_to_dest'])
            new_v_index = np.argmin(self.temp_db.status_dict['time_to_dest'])

            print('next_step_time',next_step_time)

            if next_step_time == 0 and sum(self.temp_db.status_dict['demand']) == 0:
                masked_array = np.ma.masked_where(np.array(self.temp_db.status_dict['time_to_dest'])==0, np.array(self.temp_db.status_dict['time_to_dest']))
                next_step_time = np.min(masked_array)
                new_v_index = np.argmin(masked_array)

            if next_step_time == 0 and self.v_did_sth == False:
                print(self.temp_db.cur_v_index)
                new_v_index = self.temp_db.cur_v_index + 1
                print(new_v_index)

                if new_v_index >= self.temp_db.num_vehicles:
                    new_v_index = 0
                next_step_time = self.temp_db.status_dict['time_to_dest'][self.temp_db.cur_v_index]

            if next_step_time != 0:
                [v.travel_period(next_step_time) for v in self.temp_db.base_groups['vehicles'] if self.temp_db.status_dict['v_free'][v.v_index] == 1]

        self.temp_db.cur_v_index = new_v_index
        self.v_did_sth = False
        print(np.sum(np.array(self.temp_db.status_dict['v_free'])-1)*10000)
        return sum(self.temp_db.status_dict['demand']) + sum(self.temp_db.status_dict['time_to_dest']) == np.sum(np.array(self.temp_db.status_dict['v_free'])-1)*-10000

    #def finish_episode(self):
        # force return to depots for tsp






