'''

'''
import numpy as np
from main.simulation.restrictions import RestrValueObject
from main.simulation.common_sim_func import param_interpret, random_coordinates


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
    real_charge = self.range_restr.add_value(self.charge_rate)

def street_distance(self, direction):
    return np.sum(np.abs(direction))

def arial_distance(self, direction):
    return np.linalg.norm(direction)


# Base Battery:
# ----------------------------------------------------------------------------------------------------------------

class BaseBatteryClass(RestrValueObject):

    def __init__(self, battery, name, obj_index, index_type, temp_db, speed):

        super().__init__(name, obj_index, index_type, temp_db, rate=speed)

        self.battery = battery
        self.charge_to_distance = 0.5 ####################################################################################################################

    def add_value(self, value):
        new_charge = self.battery.check_add_value(value)
        new_distance = self.charge_to_distance*new_charge
        value = min(new_distance, self.temp_db.status_dict['in_time_'+self.name][self.obj_index])
        new_value, restr_signal = self.restriction.add_value(self.temp_db.status_dict[self.name][self.obj_index], value)
        self.update(new_value, restr_signal)
        self.battery.add_value((new_value / self.charge_to_distance) - new_charge)
        return new_value

    def subtract_value(self, value):
        new_charge = self.battery.check_subtract_value(value)
        new_charge = self.charge_to_distance*new_charge
        value = min(new_distance, self.temp_db.status_dict['in_time_'+self.name][self.obj_index])
        new_value, restr_signal = self.restriction.subtract_value(self.temp_db.status_dict[self.name][self.obj_index], value)
        self.update(new_value, restr_signal)
        self.battery.subtract_value(new_charge - (new_value / self.charge_to_distance))
        return new_value

    def check_add_value(self, value):
        new_charge = self.battery.check_add_value(value)
        new_charge = self.charge_to_distance * new_charge
        value = min(new_distance, self.temp_db.status_dict['in_time_'+self.name][self.obj_index])
        new_value, restr_signal = self.restriction.add_value(self.temp_db.status_dict[self.name][self.obj_index], value)
        return new_value - self.temp_db.status_dict[self.name][self.obj_index]

    def check_subtract_value(self, value):
        new_charge = self.battery.check_subtract_value(value)
        new_charge = self.charge_to_distance * new_charge
        value = min(new_distance, self.temp_db.status_dict['in_time_'+self.name][self.obj_index])
        new_value, restr_signal = self.restriction.subtract_value(self.temp_db.status_dict[self.name][self.obj_index], value)
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

        # Intepret parameter dict:
        #for key in v_params.keys(): v_params[key] = param_interpret(v_params[key])

        # Init parameter based on parameter dict
        self.v_name = v_params['v_name']
        self.range_type = v_params['range_type']
        self.travel_type = v_params['travel_type']
        self.cargo_type = v_params['cargo_type']
        self.v_loadable = v_params['loadable']
        self.v_weight = param_interpret(v_params['weight'])

        # Create items as restricted value:
        if v_params['range_type'] == 'simple':
            self.go = simple_go
            self.recharge = simple_recharge
            self.range_restr = RestrValueObject('range', v_index, 'vehicle', temp_db, v_params['max_range'], 0, v_params['max_range'], v_params['speed'])

        elif v_params['range_type'] == 'battery':
            self.go = battery_go
            self.recharge = battery_recharge
            self.range_restr = BaseBatteryClass(
                RestrValueObject('battery', v_index, 'vehicle', temp_db, v_params['max_charge'], 0, v_params['init_charge'], v_params['charge_rate']), 
                'range', v_index, 'vehicle', temp_db, v_params['speed']
            )

        else:
            raise Exception("range_type was set to '{}', but has to be: 'simple', 'battery'".format(v_params['range_type']))

        # Create cargo as restricted value:
        if v_params['cargo_type'] == 'standard+extra':
            self.items = RestrValueObject('items', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.cargo = RestrValueObject('cargo', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.loaded_v = RestrValueObject('loaded_v', v_index, 'vehicle', temp_db, v_params['max_v_cap'], 0, 0, v_params['v_rate'])
            self.is_truck = True
            
        elif v_params['cargo_type'] == 'standard+including':
            self.items = RestrValueObject('items', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.cargo = RestrValueObject('cargo', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.loaded_v = RestrValueObject('loaded_v', v_index, 'vehicle', temp_db, None, None, None, v_params['v_rate'])
            self.is_truck = True

        elif v_params['cargo_type'] == 'standard':
            self.items = RestrValueObject('items', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
            self.cargo = RestrValueObject('cargo', v_index, 'vehicle', temp_db, v_params['max_cargo'], 0, v_params['init_cargo'], v_params['cargo_rate'])
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


    def step(self, time):
        # v_acts[0] = function that calcs a value
        # v_acts[1] = keys list
        # v_acts[2] = index list for the keys
        # v_acts[3] = function that takes an action
        v_acts = self.temp_db.v_acts[self.v_index]
        should_value = v_acts[0]()
        real_value, min_rate_constant = v_acts[1]()

        if action_value != 0:
            real_value = v_acts[2](real_value, should_value)

        if should_value-real_value == 0:
            self.temp_db.status_dict['time_till_fin'][self.v_index] = 0
        else:
            self.temp_db.status_dict['time_till_fin'][self.v_index] = (should_value-real_value) / min_rate_constant
                    
            



# Base Vehicle Creator:
# ----------------------------------------------------------------------------------------------------------------

class BaseVehicleCreator:

    def __init__(self, temp_db, v_params_list, VehicleClass=BaseVehicleClass):

        self.temp_db = temp_db
        self.v_params_list = v_params_list

        self.temp_db.num_vehicles = sum([np.max(v_params['num']) for v_params in v_params_list])
        self.temp_db.min_max_dict['v_type'] = np.array([0, len(v_params_list) - 1])

    def create(self):

        v_index = 0
        v_type = 0

        for v_params in v_params_list:
            for i in range(param_interpret(v_params['num'])):
                vehicle = VehicleClass(self.temp_db, v_index, v_type, v_params)
                self.temp_db.add_vehicle(vehicle, v_index, v_type)
                v_index +=1
            v_type += 1
