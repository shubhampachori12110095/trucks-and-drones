'''

'''
import numpy as np

from trucks_and_drones.simulation.restrictions import RestrValueObject, is_None, is_not_None, none_add, none_subtract
from trucks_and_drones.simulation.common_sim_func import param_interpret, random_coordinates


''' VEHICLE PARAMETER 
# number of vehicles:
num: (int, list, tuple, np.ndarray) = 1,
# vehicle name:
v_name: str = 'vehicle', # alt: 'vehicle', 'truck', 'drone', 'robot'
# loadable:
loadable: bool = False,
weight: (NoneType, int, list, tuple, np.ndarray) = 0,
# range or battery:
range_type: str = 'simple', # alt: 'simple', 'battery'
max_range: (NoneType, int, list, tuple, np.ndarray) = None,
max_charge: (NoneType, int, list, tuple, np.ndarray) = None,
init_charge: (str, NoneType, int, list, tuple, np.ndarray) = None,
charge_rate: (str, NoneType, int, list, tuple, np.ndarray) = None,
# travel:
travel_type: str = 'street', # alt: 'street', arial
speed: (int, list, tuple, np.ndarray) = 1,
# cargo:
cargo_type: str = 'standard', # alt: 'standard', 'standard+extra', 'standard+including'
max_cargo: (NoneType, int, list, tuple, np.ndarray) = None,
init_cargo: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
cargo_rate: (NoneType, int, list, tuple, np.ndarray) = None,
# vehicle capacity:
max_v_cap: (NoneType, int, list, tuple, np.ndarray) = 0,
v_rate: (NoneType, int, list, tuple, np.ndarray) = 0,
# visualization:
symbol: (str, NoneType) = 'circle', # 'triangle-up', 'triangle-down' 'rectangle'
color: (str, NoneType, int, list, tuple, np.ndarray) = 'red',
'''

# Vehicle Class Functions:
# ----------------------------------------------------------------------------------------------------------------


def simple_go(self, distance):
    if self.range_restr.max_value is None:
        return distance
    distance = self.range_restr.subtract_value(distance)
    return distance


def simple_recharge(self):
    self.range_restr.set_to_max()


def battery_go(self, distance):
    real_discharge = self.range_restr.subtract_value(distance * self.charge_rate)
    distance = real_discharge / self.charge_rate
    return distance


def battery_recharge(self):
    self.range_restr.add_value(self.charge_rate)


def street_distance(direction):
    return np.sum(np.abs(direction))


def arial_distance(direction):
    return np.linalg.norm(direction)


# Base Battery:
# ----------------------------------------------------------------------------------------------------------------

class BaseBatteryClass(RestrValueObject):

    def __init__(self, battery, name, obj_index, index_type, temp_db, speed):

        super().__init__(name, obj_index, index_type, temp_db, rate=speed)

        self.battery = battery
        self.charge_to_distance = 0.5  ############################################

    def calc_time(self, distance):
        time = np.nanmax(np.array([0, self.rate], dtype=np.float))*distance
        return np.nanmin(time, self.battery.calc_time(distance / self.charge_to_distance))

    def add_value(self, charge):
        new_charge = self.battery.check_add_value(charge)
        distance = self.charge_to_distance*new_charge

        if is_not_None(self.temp_db.status_dict['in_time_'+self.name][self.obj_index]):
           distance = min(distance, self.temp_db.status_dict['in_time_'+self.name][self.obj_index])
        
        new_value, restr_signal = self.restriction.add_value(self.temp_db.status_dict[self.name][self.obj_index], distance)
        self.update(new_value, restr_signal)
        self.battery.add_value((new_value / self.charge_to_distance) - new_charge)
        return new_value


    def subtract_value(self, distance):
        new_charge = self.battery.check_subtract_value(distance/self.charge_to_distance)
        distance = self.charge_to_distance*new_charge
        
        if is_not_None(self.temp_db.status_dict['in_time_'+self.name][self.obj_index]):
            distance = min(distance, self.temp_db.status_dict['in_time_'+self.name][self.obj_index])

        new_value, restr_signal = self.restriction.subtract_value(self.temp_db.status_dict[self.name][self.obj_index], distance)
        self.update(new_value, restr_signal)
        self.battery.subtract_value(new_charge - (new_value / self.charge_to_distance))
        return new_value


    def check_add_value(self, value, in_time=True):
        new_charge = self.battery.check_add_value(value)
        distance = self.charge_to_distance * new_charge

        if is_not_None(self.temp_db.status_dict['in_time_'+self.name][self.obj_index]) and in_time:
            distance = min(distance, self.temp_db.status_dict['in_time_'+self.name][self.obj_index])
        
        new_value, restr_signal = self.restriction.add_value(self.temp_db.status_dict[self.name][self.obj_index], distance)
        
        if is_None(self.temp_db.status_dict[self.name][self.obj_index]):
            return new_value
        return new_value - self.temp_db.status_dict[self.name][self.obj_index]


    def check_subtract_value(self, distance, in_time=True):
        new_charge = self.battery.check_subtract_value(distance/self.charge_to_distance)
        distance = self.charge_to_distance * new_charge
        
        if is_not_None(self.temp_db.status_dict['in_time_'+self.name][self.obj_index]) and in_time:
            distance = min(distance, self.temp_db.status_dict['in_time_'+self.name][self.obj_index])
        
        new_value, restr_signal = self.restriction.subtract_value(self.temp_db.status_dict[self.name][self.obj_index], distance)
        
        if is_None(self.temp_db.status_dict[self.name][self.obj_index]):
            return new_value
        return self.temp_db.status_dict[self.name][self.obj_index] - new_value


# Base Vehicle Class:
# ----------------------------------------------------------------------------------------------------------------

class BaseVehicleClass:
    '''
    '''
    def __init__(self, temp_db, v_index, v_type, v_params, BatteryClass=BaseBatteryClass):

        # Init:
        self.temp_db = temp_db
        self.v_index = v_index
        self.v_type = v_type

        # Init parameter based on parameter dict
        self.v_name = v_params['v_name']
        self.range_type = v_params['range_type']
        self.travel_type = v_params['travel_type']
        self.cargo_type = v_params['cargo_type']
        self.v_loadable = v_params['loadable']
        self.v_weight = param_interpret(v_params['weight'])

        self.temp_db.min_max_dict['v_weight'] = np.append(self.temp_db.min_max_dict['v_weight'], v_params['weight'])

        # Create items as restricted value:
        if v_params['range_type'] == 'simple':
            self.go = simple_go
            self.recharge = simple_recharge
            self.range_restr = RestrValueObject('v_range', v_index, 'vehicle', temp_db, v_params['max_range'], 0, v_params['max_range'], v_params['speed'])

        elif v_params['range_type'] == 'battery':
            self.go = battery_go
            self.recharge = battery_recharge
            self.range_restr = BatteryClass(
                RestrValueObject('battery', v_index, 'vehicle', temp_db, v_params['max_charge'], 0, v_params['init_charge'], v_params['charge_rate']), 
                'v_range', v_index, 'vehicle', temp_db, v_params['speed']
            )

        else:
            raise Exception("range_type was set to '{}', but has to be: 'simple', 'battery'".format(v_params['range_type']))

        # Create cargo as restricted value:
        if v_params['cargo_type'] == 'standard+extra':
            self.v_items = RestrValueObject('v_items', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.v_cargo = RestrValueObject('v_cargo', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.loaded_v = RestrValueObject('loaded_v', v_index, 'vehicle', temp_db, v_params['max_v_cap'], 0, 0, v_params['v_rate'])
            self.is_truck = True
            
        elif v_params['cargo_type'] == 'standard+including':
            self.v_items = RestrValueObject('v_items', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.v_cargo = RestrValueObject('v_cargo', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.loaded_v = RestrValueObject('loaded_v', v_index, 'vehicle', temp_db, None, None, None, v_params['v_rate'])
            self.is_truck = True

        elif v_params['cargo_type'] == 'standard':
            self.v_items = RestrValueObject('v_items', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.v_cargo = RestrValueObject('v_cargo', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.loaded_v = RestrValueObject('loaded_v', v_index, 'vehicle', temp_db, 0, 0, 0, 0)
            self.is_truck = False

        else:
            raise Exception("cargo_type was set to '{}', but has to be: 'standard+extra', 'standard+including', 'only_standard'".format(v_params['cargo_type']))

        # Init travel functions:
        if v_params['travel_type'] == 'street':
            self.calc_distance = street_distance
        elif v_params['travel_type'] == 'arial':
            self.calc_distance = arial_distance
        else:
            raise Exception("travel_type was set to '{}', but has to be: 'street', 'arial'".format(v_params['travel_type']))


    def take_action(self, calc_time=False):
        
        if len(self.temp_db.actions_list[self.v_index]) > 0:
            
            action_pack = self.temp_db.actions_list[self.v_index][0]
            #print('actions', action_pack)

            
            if action_pack[0] == 'move':
                self.v_move(calc_time=calc_time)
            elif action_pack[0] == 'load_v':
                self.v_load_v(action_pack[1], calc_time=calc_time)
            elif action_pack[0] == 'unload_v':
                self.v_unload_v(action_pack[1], action_pack[2], calc_time=calc_time)
            elif action_pack[0] == 'load_i':
                self.v_load_items(action_pack[1], action_pack[2], calc_time=calc_time)
            elif action_pack[0] == 'unload_i':
                self.v_unload_items(action_pack[1], action_pack[2], calc_time=calc_time)

        else:
            self.temp_db.time_till_fin[self.v_index] = None


    def v_move(self, calc_time=False):

        direction = self.temp_db.status_dict['v_dest'][self.v_index] - self.temp_db.status_dict['v_coord'][self.v_index]
        distance = self.calc_distance(direction)

        if np.round(np.array([distance]), 3) == 0:
            self.temp_db.status_dict['v_coord'][self.v_index] = self.temp_db.status_dict['v_dest'][self.v_index]
            # self.temp_db.actions_list[self.v_index].pop(0)
            # self.take_action(calc_time=True)
            return

        if bool(self.temp_db.status_dict['v_stuck'][self.v_index]):
            # self.temp_db.actions_list[self.v_index].pop(0)
            # self.take_action(calc_time=True)
            return

        if not calc_time:

            real_distance = self.range_restr.check_subtract_value(distance)

            if real_distance != 0:
                self.range_restr.subtract_value(distance)
                self.temp_db.status_dict['v_coord'][self.v_index] = (
                    direction * (real_distance/distance) + self.temp_db.status_dict['v_coord'][self.v_index]
                )
                if self.is_truck:
                    self.temp_db.past_coord_not_transportable_v[self.v_index].append(np.copy(self.temp_db.status_dict['v_coord'][self.v_index]))
                else:
                    self.temp_db.past_coord_transportable_v[self.v_index].append(np.copy(self.temp_db.status_dict['v_coord'][self.v_index]))

                
                for i in self.temp_db.v_transporting_v[self.v_index]:
                    self.temp_db.status_dict['v_coord'][i] = self.temp_db.status_dict['v_coord'][self.v_index]
                    self.temp_db.past_coord_transportable_v[i].append(np.copy(
                        self.temp_db.status_dict['v_coord'][self.v_index]))

            if np.round(real_distance - distance, 3) == 0:
                # self.temp_db.actions_list[self.v_index].pop(0)
                # self.take_action(calc_time=True)
                return

            elif self.temp_db.cur_time_frame != 0:
                self.temp_db.time_till_fin[self.v_index] = (real_distance / self.temp_db.cur_time_frame) * (
                            distance - real_distance)
                if real_distance == 0:
                    self.temp_db.status_dict['v_stuck'][self.v_index] = 1

            else:
                self.temp_db.time_till_fin[self.v_index] = np.nanmax(
                    self.range_restr.calc_time(distance - real_distance), 0
                )

        else:
            self.temp_db.time_till_fin[self.v_index] = np.nanmax(
                self.range_restr.calc_time(distance), 0
            )


    def v_load_v(self, v_j, item_amount=None, calc_time=False):

        cargo_i = self.v_cargo
        items_i = self.v_items
        loaded_v_i = self.loaded_v
        
        cargo_j = self.temp_db.restr_dict['v_cargo'][v_j]
        items_j = self.temp_db.restr_dict['v_items'][v_j]
        weight_j = self.temp_db.constants_dict['v_weight'][v_j]

        if (not bool(self.temp_db.constants_dict['v_loadable'][v_j])
                #or items_i.max_restr - items_i.cur_value() < cargo_j.cur_value()
                or not bool(self.temp_db.status_dict['v_free'][v_j])
                or not self.is_truck
                or loaded_v_i.rate == 0
                or not loaded_v_i.check_add_value(1, in_time=False) == 1
            ):
            self.temp_db.actions_list[self.v_index].pop(0)
            self.take_action(calc_time=True)
            return

        item_amount = np.nanmin(np.array([
            cargo_j.cur_value(),
            none_subtract(cargo_i.check_add_value(none_add(item_amount, weight_j), in_time=False), weight_j),
            items_i.check_add_value(item_amount, in_time=False),
            cargo_j.check_subtract_value(item_amount, in_time=False),
            items_j.check_subtract_value(item_amount, in_time=False)
        ], dtype=np.float))
        
        if item_amount == 0 and np.nanmin(
                np.array([items_j.cur_value(), none_subtract(items_i.max_restr, items_i.cur_value())], dtype=np.float)
            )!= 0:
            self.temp_db.actions_list[self.v_index].pop(0)
            self.take_action(calc_time=True)
            return

        if not calc_time and loaded_v_i.check_add_value(1) == 1:
                
            real_item_amount = np.nanmin(np.array([
                none_subtract(cargo_i.check_add_value(none_add(item_amount, weight_j)), weight_j),
                items_i.check_add_value(item_amount),
                cargo_j.check_subtract_value(item_amount),
                items_j.check_subtract_value(item_amount)
            ], dtype=np.float))

            self.temp_db.signals_dict['cargo_loss'][self.v_index] += abs(items_j.cur_value() - real_item_amount)

            if real_item_amount > 0:
                cargo_i.add_value(real_item_amount + weight_j)
                items_i.add_value(real_item_amount)
                cargo_j.subtract_value(real_item_amount)
                items_j.subtract_value(real_item_amount)

            if np.round(real_item_amount, 3) == np.round(item_amount, 3):
                cargo_i.round_cur_value()
                items_i.round_cur_value()
                cargo_j.round_cur_value()
                items_j.round_cur_value()

                loaded_v_i.add_value(1)
                self.temp_db.v_transporting_v[self.v_index].append(v_j)
                self.temp_db.status_dict['v_free'][v_j] = 0
                self.temp_db.status_dict['v_to_n'][v_j] = None

                
                self.temp_db.actions_list[self.v_index].pop(0)
                self.take_action(calc_time=True)
                return
            
            else:
                self.temp_db.time_till_fin[self.v_index] = np.nanmax(np.array([
                    cargo_i.calc_time(item_amount-real_item_amount),
                    items_i.calc_time(item_amount-real_item_amount),
                    cargo_j.calc_time(item_amount-real_item_amount),
                    items_j.calc_time(item_amount-real_item_amount),
                    loaded_v_i.calc_time(1)
                ], dtype=np.float))

        else:
            self.temp_db.time_till_fin[self.v_index] = np.nanmax(np.array([
                cargo_i.calc_time(item_amount + weight_j),
                items_i.calc_time(item_amount),
                cargo_j.calc_time(item_amount),
                items_j.calc_time(item_amount),
                loaded_v_i.calc_time(1)
            ], dtype=np.float))


    def v_unload_v(self, v_j, item_amount=None, calc_time=False):

        cargo_i = self.v_cargo
        items_i = self.v_items
        loaded_v_i = self.loaded_v
        
        cargo_j = self.temp_db.restr_dict['v_cargo'][v_j]
        items_j = self.temp_db.restr_dict['v_items'][v_j]
        weight_j = self.temp_db.constants_dict['v_weight'][v_j]

        possible_item_amount = np.nanmin(np.array([
            none_subtract(cargo_i.check_subtract_value(none_add(item_amount, weight_j), in_time=False), weight_j),
            items_i.check_subtract_value(item_amount, in_time=False),
            cargo_j.check_add_value(item_amount, in_time=False),
            items_j.check_add_value(item_amount, in_time=False)
        ], dtype=np.float))

        if not item_amount is None:
            action_error = abs(item_amount-possible_item_amount)
        item_amount = possible_item_amount

        if not calc_time and loaded_v_i.check_subtract_value(1) == 1:
        
            real_item_amount = np.nanmin(np.array([
                none_subtract(cargo_i.check_subtract_value(none_add(item_amount, weight_j)), weight_j),
                items_i.check_subtract_value(item_amount),
                cargo_j.check_add_value(item_amount),
                items_j.check_add_value(item_amount),
            ], dtype=np.float))

            if real_item_amount > 0:
                cargo_i.subtract_value(real_item_amount + weight_j)
                items_i.subtract_value(real_item_amount)
                cargo_j.add_value(real_item_amount)
                items_j.add_value(real_item_amount)

            if np.round(real_item_amount, 3) == np.round(item_amount, 3):
                cargo_i.round_cur_value()
                items_i.round_cur_value()
                cargo_j.round_cur_value()
                items_j.round_cur_value()
                loaded_v_i.subtract_value(1)
                
                self.temp_db.v_transporting_v[self.v_index].pop(self.temp_db.v_transporting_v[self.v_index].index(v_j))
                self.temp_db.status_dict['v_free'][v_j] = 1
                
                self.temp_db.actions_list[self.v_index].pop(0)
                self.take_action(calc_time=True)
                return
            
            else:
                self.temp_db.time_till_fin[self.v_index] = np.nanmax(np.array([
                    cargo_i.calc_time(item_amount-real_item_amount),
                    items_i.calc_time(item_amount-real_item_amount),
                    cargo_j.calc_time(item_amount-real_item_amount),
                    items_j.calc_time(item_amount-real_item_amount),
                    loaded_v_i.calc_time(1)
                ], dtype=np.float))

        else:
            self.temp_db.time_till_fin[self.v_index] = np.nanmax(np.array([
                cargo_i.calc_time(item_amount + weight_j),
                items_i.calc_time(item_amount),
                cargo_j.calc_time(item_amount),
                items_j.calc_time(item_amount),
                loaded_v_i.calc_time(1)
            ], dtype=np.float))


    def v_unload_items(self, n_j, item_amount=None, calc_time=False):

        cargo_i = self.v_cargo
        items_i = self.v_items
        
        items_j = self.temp_db.restr_dict['n_items'][n_j]

        possible_item_amount = np.nanmin(np.array([
            cargo_i.check_subtract_value(item_amount, in_time=False),
            items_i.check_subtract_value(item_amount, in_time=False),
            items_j.check_subtract_value(item_amount, in_time=False),
        ], dtype=np.float))

        if not item_amount is None:
            action_error = abs(item_amount-possible_item_amount)
        item_amount = possible_item_amount

        if item_amount == 0:
            self.temp_db.status_dict['n_waiting'][n_j] = 0
            
            # self.temp_db.actions_list[self.v_index].pop(0)
            # self.take_action(calc_time=True)
            return

        if not calc_time:

            real_item_amount = np.nanmin(np.array([
                cargo_i.check_subtract_value(item_amount),
                items_i.check_subtract_value(item_amount),
                items_j.check_subtract_value(item_amount),
            ], dtype=np.float))
        
            if real_item_amount > 0:
                cargo_i.subtract_value(real_item_amount)
                items_i.subtract_value(real_item_amount)
                items_j.subtract_value(real_item_amount)

            if np.round(real_item_amount, 3) == np.round(item_amount, 3):
                cargo_i.round_cur_value()
                items_i.round_cur_value()
                items_j.round_cur_value()

                self.temp_db.status_dict['n_waiting'][n_j] = 0
                
                self.temp_db.actions_list[self.v_index].pop(0)
                self.take_action(calc_time=True)
                return

            else:
                self.temp_db.time_till_fin[self.v_index] = np.nanmax(np.array([
                    cargo_i.calc_time(item_amount - real_item_amount),
                    items_i.calc_time(item_amount - real_item_amount),
                    items_j.calc_time(item_amount - real_item_amount),
                ], dtype=np.float))

        else:
            self.temp_db.time_till_fin[self.v_index] = np.nanmax(np.array([
                cargo_i.calc_time(item_amount),
                items_i.calc_time(item_amount),
                items_j.calc_time(item_amount),
            ], dtype=np.float))

    def v_load_items(self, n_j, item_amount=None, calc_time=False):

        cargo_i = self.v_cargo
        items_i = self.v_items
        
        items_j = self.temp_db.restr_dict['n_items'][n_j]

        possible_item_amount = np.nanmin(np.array([
                np.sum(self.temp_db.customers(self.temp_db.status_dict['n_items'])[0]),
                cargo_i.check_add_value(item_amount, in_time=False),
                items_i.check_add_value(item_amount, in_time=False),
                items_j.check_subtract_value(item_amount, in_time=False),
        ], dtype=np.float))

        if not item_amount is None:
            action_error = abs(item_amount-possible_item_amount)
        item_amount = possible_item_amount

        if item_amount == 0:
            self.temp_db.actions_list[self.v_index].pop(0)
            self.take_action(calc_time=True)
            return        

        if not calc_time:

            real_item_amount = np.nanmin(np.array([
                cargo_i.check_add_value(item_amount),
                items_i.check_add_value(item_amount),
                items_j.check_subtract_value(item_amount),
            ], dtype=np.float))

            if real_item_amount > 0:
                cargo_i.add_value(real_item_amount)
                items_i.add_value(real_item_amount)
                items_j.subtract_value(real_item_amount)

            if np.round(real_item_amount, 3) == np.round(item_amount, 3):
                cargo_i.round_cur_value()
                items_i.round_cur_value()
                items_j.round_cur_value()

                self.temp_db.actions_list[self.v_index].pop(0)
                self.take_action(calc_time=True)
                return

            else:
                self.temp_db.time_till_fin[self.v_index] = np.nanmax(np.array([
                    cargo_i.calc_time(item_amount - real_item_amount),
                    items_i.calc_time(item_amount - real_item_amount),
                    items_j.calc_time(item_amount - real_item_amount),
                ], dtype=np.float))

        else:
            self.temp_db.time_till_fin[self.v_index] = np.nanmax(np.array([
                cargo_i.calc_time(item_amount),
                items_i.calc_time(item_amount),
                items_j.calc_time(item_amount),
            ], dtype=np.float))


# Base Vehicle Creator:
# ----------------------------------------------------------------------------------------------------------------

class BaseVehicleCreator:

    def __init__(self, v_params_list, temp_db, VehicleClass=BaseVehicleClass):

        self.temp_db = temp_db
        self.v_params_list = v_params_list

        self.temp_db.num_vehicles = sum([np.max(v_params['num']) for v_params in v_params_list])

        self.VehicleClass = VehicleClass

    def create(self):

        v_index = 0
        v_type = 0

        for v_params in self.v_params_list:
            for i in range(param_interpret(v_params['num'])):
                vehicle = self.VehicleClass(self.temp_db, v_index, v_type, v_params)
                self.temp_db.add_vehicle(vehicle, v_index, v_type)
                v_index +=1
            self.temp_db.vehicle_visuals.append([v_params['symbol'], v_params['color']])
            v_type += 1

        self.temp_db.min_max_dict['v_type'] = np.array([0, len(self.v_params_list) - 1])
