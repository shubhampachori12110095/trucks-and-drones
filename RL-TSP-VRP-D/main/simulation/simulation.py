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


# Action Functions (used by vehicle objects):
# ----------------------------------------------------------------------------------------------------------------

def v_load_v(self, v_j, item_amount=None):

    cargo_i = self.cargo
    items_i = self.items
    loaded_v_i = self.loaded_v
    
    cargo_j = self.temp_db.restr_dict['cargo'][v_j]
    items_j = self.temp_db.restr_dict['items'][v_j]
    weight_j = self.temp_db.constants_dict['weight'][v_j]

    if items_i.max_restr - items_i.cur_value() < cargo_j.cur_value() or self.temp_db.constants_dict['loadable'][v_j] == 0:
        self.temp_db.actions_list[self.v_index].pop(0)
        
        if len(self.temp_db.actions_list[self.v_index]) == 0:
            self.temp_db.time_till_fin[self.v_index] = None
        else:
            self.temp_db.time_till_fin[self.v_index] = 0
        return

    if loaded_v_i.cur_value() is None or bool(loaded_v_i.check_add_value(1)):
        item_amount = items_j.cur_value()
    
        real_item_amount = item_amount - min(
            cargo_i.check_add_value(item_amount + weight) - weight,
            items_i.check_add_value(item_amount),
            cargo_j.check_subtract_value(item_amount),
            items_j.check_subtract_value(item_amount)
        )

        if real_item_amount > 0:
            cargo_i.add_value(real_item_amount + weight)
            items_i.add_value(real_item_amount)
            cargo_j.subtract_value(real_item_amount)
            items_j.subtract_value(real_item_amount)

        if real_item_amount == item_amount:
            loaded_v_i.add_value(1):
            self.temp_db.v_transporting_v[self.v_index].append(v_j)
            self.temp_db.actions_list[self.v_index].pop(0)
            
            if len(self.temp_db.actions_list[self.v_index]) == 0:
                self.temp_db.time_till_fin[self.v_index] = None
            else:
                self.temp_db.time_till_fin[self.v_index] = 0
        
        else:
            self.temp_db.time_till_fin[self.v_index] = (real_item_amount / self.temp_db.cur_time_frame) * (item_amount - real_item_amount)

    else:
        self.temp_db.time_till_fin[self.v_index] = 1 / loaded_v_i.rate


def v_unload_v(self, v_j, item_amount=None):

    cargo_i = self.cargo
    items_i = self.items
    loaded_v_i = self.loaded_v
    
    cargo_j = self.temp_db.restr_dict['cargo'][v_j]
    items_j = self.temp_db.restr_dict['items'][v_j]
    weight_j = self.temp_db.constants_dict['weight'][v_j]

    if loaded_v_i.cur_value() is None or bool(loaded_v_i.check_subtract_value(1)):
        
        if item_amount is None:
            item_amount = min(items_i.cur_value(), items_j.max_restr)
    
        real_item_amount = min(
            cargo_i.check_subtract_value(item_amount + weight) - weight,
            items_i.check_subtract_value(item_amount),
            cargo_j.check_add_value(item_amount),
            items_j.check_add_value(item_amount),
        )

        if real_item_amount > 0:
            cargo_i.subtract_value(real_item_amount + weight)
            items_i.subtract_value(real_item_amount)
            cargo_j.add_value(real_item_amount)
            items_j.add_value(real_item_amount)


        if real_item_amount == item_amount:
            loaded_v_i.subtract_value(1):
            self.temp_db.v_transporting_v[self.v_index].pop(self.temp_db.v_transporting_v[self.v_index].index(v_j))
            self.temp_db.actions_list[self.v_index].pop(0)
            
            if len(self.temp_db.actions_list[self.v_index]) == 0:
                self.temp_db.time_till_fin[self.v_index] = None
            else:
                self.temp_db.time_till_fin[self.v_index] = 0
        
        else:
            self.temp_db.time_till_fin[self.v_index] = (real_item_amount / self.temp_db.cur_time_frame) * (item_amount - real_item_amount)

    else:
        self.temp_db.time_till_fin[self.v_index] = 1 / loaded_v_i.rate


def v_unload_items(self, n_j, item_amount=None):

    cargo_i = self.cargo
    items_i = self.items
    
    items_j = self.temp_db.restr_dict['n_items'][n_j]

    if item_amount is None:
        item_amount = min(items_i.cur_value(), items_j.cur_value())

    real_item_amount = min(
        cargo_i.check_subtract_value(item_amount)
        items_i.check_subtract_value(item_amount)
        items_j.check_subtract_value(item_amount)
    )

    if real_item_amount > 0:
        cargo_i.subtract_value(real_item_amount)
        items_i.subtract_value(real_item_amount)
        items_j.subtract_value(real_item_amount)

    if real_item_amount == item_amount:
        self.temp_db.actions_list[self.v_index].pop(0)
        
        if len(self.temp_db.actions_list[self.v_index]) == 0:
            self.temp_db.time_till_fin[self.v_index] = None
        else:
            self.temp_db.time_till_fin[self.v_index] = 0

    else:
        self.temp_db.time_till_fin[self.v_index] = (real_cargo_amount / self.temp_db.cur_time_frame) * (item_amount - real_item_amount)


def v_load_items(self, n_j, item_amount=None):

    cargo_i = self.cargo
    items_i = self.items
    
    items_j = self.temp_db.restr_dict['n_items'][n_j]

    if item_amount is None:
        item_amount = min(items_i.cur_value(), items_j.cur_value())

    real_item_amount = min(
        cargo_i.check_add_value(item_amount)
        items_i.check_add_value(item_amount)
        items_j.check_subtract_value(item_amount)
    )

    if real_item_amount > 0:
        cargo_i.add_value(real_item_amount)
        items_i.add_value(real_item_amount)
        items_j.subtract_value(real_item_amount)

    if real_item_amount == item_amount:
        self.temp_db.actions_list[self.v_index].pop(0)
        
        if len(self.temp_db.actions_list[self.v_index]) == 0:
            self.temp_db.time_till_fin[self.v_index] = None
        else:
            self.temp_db.time_till_fin[self.v_index] = 0

    else:
        self.temp_db.time_till_fin[self.v_index] = (real_cargo_amount / self.temp_db.cur_time_frame) * (item_amount - real_item_amount)


# Base Simulator Class:
# ----------------------------------------------------------------------------------------------------------------

class BaseSimulator:

    def __init__(self, temp_db, vehicle_creator, node_creator):

        [setattr(self, k, v) for k, v in sim_param.items()]

        self.temp_db = temp_db
        self.vehicle_creator = vehicle_creator
        self.node_creator = node_creator


    def reset_simulation(self):

        self.temp_db.init_db()
        self.node_creator.create()
        self.vehicle_creator.create()
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






