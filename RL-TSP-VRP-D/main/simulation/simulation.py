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
    
def v_move(self, _, _, calc_time=False):

    direction = self.temp_db.status_dict['v_dest'][self.v_index] - self.temp_db.status_dict['v_coord'][self.v_index]
    distance = self.calc_distance(direction)

    real_distance = self.range_obj.check_subtract_value(distance)
    
    if real_distance != 0 and not calc_time:
        self.range_obj.subtract_value(distance)
        self.temp_db.status_dict['v_coord'][self.v_index] = direction * (real_distance/distance) + self.temp_db.status_dict['v_coord'][self.v_index]
        
        for i in self.temp_db.v_transporting_v[self.v_index]:
            self.temp_db.status_dict['v_coord'][i] = self.temp_db.status_dict['v_coord'][self.v_index]
    
    if real_distance == distance and not calc_time:

        self.temp_db.actions_list[self.v_index].pop(0)
        
        if len(self.temp_db.actions_list[self.v_index]) == 0:
            self.temp_db.time_till_fin[self.v_index] = None
        else:
            self.temp_db.time_till_fin[self.v_index] = 0

    else:
        self.temp_db.time_till_fin[self.v_index] = (real_distance / self.temp_db.cur_time_frame) * (distance - real_distance)

    if calc_time:
        self.temp_db.time_till_fin[self.v_index] = self.temp_db.time_till_fin[self.v_index] + 1



def v_load_v(self, v_j, item_amount=None, calc_time=False):

    cargo_i = self.cargo
    items_i = self.items
    loaded_v_i = self.loaded_v
    
    cargo_j = self.temp_db.restr_dict['cargo'][v_j]
    items_j = self.temp_db.restr_dict['items'][v_j]
    weight_j = self.temp_db.constants_dict['weight'][v_j]

    if items_i.max_restr - items_i.cur_value() < cargo_j.cur_value() or self.temp_db.constants_dict['loadable'][v_j] == 0 or self.temp_db.status_dict['v_free'][v_j] == 0:
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

        if real_item_amount > 0 and not calc_time:
            cargo_i.add_value(real_item_amount + weight)
            items_i.add_value(real_item_amount)
            cargo_j.subtract_value(real_item_amount)
            items_j.subtract_value(real_item_amount)

        if real_item_amount == item_amount and not calc_time:
            loaded_v_i.add_value(1):
            self.temp_db.v_transporting_v[self.v_index].append(v_j)
            self.temp_db.actions_list[self.v_index].pop(0)
            self.temp_db.status_dict['v_free'][v_j] = 0
            
            if len(self.temp_db.actions_list[self.v_index]) == 0:
                self.temp_db.time_till_fin[self.v_index] = None
            else:
                self.temp_db.time_till_fin[self.v_index] = 0
        
        else:
            self.temp_db.time_till_fin[self.v_index] = (real_item_amount / self.temp_db.cur_time_frame) * (item_amount - real_item_amount)

        if calc_time:
            self.temp_db.time_till_fin[self.v_index] = self.temp_db.time_till_fin[self.v_index] + 1
    else:
        self.temp_db.time_till_fin[self.v_index] = 1 / loaded_v_i.rate


def v_unload_v(self, v_j, item_amount=None, calc_time=False):

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

        if real_item_amount > 0 and not calc_time:
            cargo_i.subtract_value(real_item_amount + weight)
            items_i.subtract_value(real_item_amount)
            cargo_j.add_value(real_item_amount)
            items_j.add_value(real_item_amount)


        if real_item_amount == item_amount and not calc_time:
            loaded_v_i.subtract_value(1):
            self.temp_db.v_transporting_v[self.v_index].pop(self.temp_db.v_transporting_v[self.v_index].index(v_j))
            self.temp_db.actions_list[self.v_index].pop(0)
            self.temp_db.status_dict['v_free'][v_j] = 1
            
            if len(self.temp_db.actions_list[self.v_index]) == 0:
                self.temp_db.time_till_fin[self.v_index] = None
            else:
                self.temp_db.time_till_fin[self.v_index] = 0
        
        else:
            self.temp_db.time_till_fin[self.v_index] = (real_item_amount / self.temp_db.cur_time_frame) * (item_amount - real_item_amount)

        if calc_time:
            self.temp_db.time_till_fin[self.v_index] = self.temp_db.time_till_fin[self.v_index] + 1

    else:
        self.temp_db.time_till_fin[self.v_index] = 1 / loaded_v_i.rate


def v_unload_items(self, n_j, item_amount=None, calc_time=False):

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

    if real_item_amount > 0 and not calc_time:
        cargo_i.subtract_value(real_item_amount)
        items_i.subtract_value(real_item_amount)
        items_j.subtract_value(real_item_amount)

    if real_item_amount == item_amount and not calc_time:
        self.temp_db.actions_list[self.v_index].pop(0)
        
        if len(self.temp_db.actions_list[self.v_index]) == 0:
            self.temp_db.time_till_fin[self.v_index] = None
        else:
            self.temp_db.time_till_fin[self.v_index] = 0

    else:
        self.temp_db.time_till_fin[self.v_index] = (real_cargo_amount / self.temp_db.cur_time_frame) * (item_amount - real_item_amount)

    if calc_time:
        self.temp_db.time_till_fin[self.v_index] = self.temp_db.time_till_fin[self.v_index] + 1


def v_load_items(self, n_j, item_amount=None, calc_time=False):

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

    if real_item_amount > 0 and not calc_time:
        cargo_i.add_value(real_item_amount)
        items_i.add_value(real_item_amount)
        items_j.subtract_value(real_item_amount)

    if real_item_amount == item_amount and not calc_time:
        self.temp_db.actions_list[self.v_index].pop(0)
        
        if len(self.temp_db.actions_list[self.v_index]) == 0:
            self.temp_db.time_till_fin[self.v_index] = None
        else:
            self.temp_db.time_till_fin[self.v_index] = 0

    else:
        self.temp_db.time_till_fin[self.v_index] = (real_cargo_amount / self.temp_db.cur_time_frame) * (item_amount - real_item_amount)

    if calc_time:
        self.temp_db.time_till_fin[self.v_index] = self.temp_db.time_till_fin[self.v_index] + 1


# Base Simulator Class:
# ----------------------------------------------------------------------------------------------------------------

class BaseSimulator:

    def __init__(self, temp_db, vehicle_creator, node_creator, auto_agent):

        self.temp_db = temp_db
        self.vehicle_creator = vehicle_creator
        self.node_creator = node_creator
        self.auto_agent = auto_agent


    def reset_simulation(self):

        self.temp_db.init_db()
        self.node_creator.create()
        self.vehicle_creator.create()
        self.temp_db.reset_db()
        self.reset_round()


    def reset_round(self):
        self.v_count = 0
        self.v_indices = np.where(self.temp_db.time_till_fin == None)
        self.num_v = len(self.v_indice)
        self.temp_db.cur_v_index = self.v_indices[self.v_count]


    def set_destination(self, coordinates=None):

        if coordinates is None:
            coordinates = self.auto_agent.find_destination()

        if coordinates is not None:
            self.temp_db.status_dict['v_dest'][self.temp_db.cur_v_index] = np.array(coordinates)
            self.temp_db.actions_list[self.temp_db.cur_v_index].append([v_move, None, None])
            print('new destination:', coordinates, 'for', self.temp_db.cur_v_index)


    def unload_vehicle(self, v_j=None, amount=None):

        if v_j is None:
            v_j = self.auto_agent.find_v_to_unload()

        if v_j is not None:
            self.temp_db.actions_list[self.temp_db.cur_v_index].append([v_unload_v, v_j, amount])
            print(v_j, 'to unload from', self.temp_db.cur_v_index, 'with', amount, 'items')


    def load_vehicle(self, v_j=None):

        if v_j is None:
            v_j = self.auto_agent.find_v_to_load()

        if v_j is not None:
            if self.temp_db.same_coord(self.temp_db.status_dict['v_coord'][v_j]):
                self.temp_db.actions_list[self.temp_db.cur_v_index].append([v_load_v, v_j, None])
                print(v_j, 'to unload to', self.temp_db.cur_v_index)


    def unload_items(self, n_j=None, amount=None):

        if n_j is None:
            n_j = self.auto_agent.find_customer()

        if n_j is not None:
            if self.temp_db.same_coord(self.temp_db.status_dict['n_coord'][n_j]):
                self.temp_db.actions_list[self.temp_db.cur_v_index].append([v_unload_items, n_j, amount])
                print(amount, 'items to unload from', self.temp_db.cur_v_index, 'to', n_j)


    def load_items(self, n_j=None, amount=None):

        if n_j is None:
            n_j = self.auto_agent.find_depot()

        if n_j is not None:
            if self.temp_db.same_coord(self.temp_db.status_dict['n_coord'][n_j]):
                self.temp_db.actions_list[self.temp_db.cur_v_index].append([v_load_items, n_j, amount])
                print(amount, 'items to load to', self.temp_db.cur_v_index, 'from', n_j)


    def recharge_range(self, vehicle_i):

        if any(self.temp_db.status_dict['v_coord'][vehicle_i] for elem in recharge_coord):
            recharge_v(self.temp_db.base_groups[vehicle_i])


    def finish_step(self):

        if self.temp_db.terminal_state():
            return True

        self.v_count += 1
        self.temp_db.cur_v_index = self.v_indices[self.v_count]

        if self.v_count == self.num_v

            self.temp_db.cur_time_frame = 1
            [restr.in_time() for restr in self.temp_db.restr_dict[key] for key in self.temp_db.restr_dict.keys()]
            [v.calc_time() for v in self.temp_db.base_groups['vehicles']]
            

            masked_array = np.ma.masked_where(self.temp_db.time_till_fin == None, self.temp_db.time_till_fin)
            self.temp_db.cur_time_frame = np.min(masked_array)
            [restr.in_time() for restr in self.temp_db.restr_dict[key] for key in self.temp_db.restr_dict.keys()]
            [v.take_action() for v in self.temp_db.base_groups['vehicles']]

            self.reset_round()

        return False
