'''

'''
import numpy as np
from main.simulation.restrictions import RestrValueObject
from main.simulation.common_sim_func import param_interpret, random_coordinates
from main.simulation.temp_database import lookup_db

# Range
# ----------------------------------------------------------------------------------------------------------------


def battery_parameter(
        range_type         = 'battery',
        max_charge         = 100,
        charge_per_step    = 50,
        discharge_per_step = 10,
        init_value         = 0,
        signal_list        = [1,1,-1],
        ):
    return {
            'range_type': range_type,
            'max_charge': max_charge,
            'charge_per_step': charge_per_step,
            'discharge_per_step': discharge_per_step,
            'init_value': init_value,
            'signal_list': signal_list,
            }


class StandardBattery:
    '''
    For battery recharge.
    '''
    def __init__(self,v_index,temp_db,battery_parameter):

        [setattr(self, k, param_interpret(v)) for k, v in battery_parameter.items()]
        
        self.SoC = RestrValueObject('battery',v_index, temp_db, self.max_charge, 0, self.init_value, self.signal_list)


    def travel_step(self,distance):
        real_discharge = self.SoC.subtract_value(distance*self.discharge_per_step)
        distance = real_discharge/self.discharge_per_step
        return distance

    def charge_travelling(self):
        real_charge = self.SoC.add_value(self.charge_per_step)


def range_parameter(
        range_type  = 'range',
        max_range   = None,
        init_value  = 0,
        signal_list = [1,1,-1],
        ):
    return {
            'range_type': range_type,
            'max_range': max_range,
            'init_value': init_value,
            'signal_list': signal_list,
            }


class StandardRange:
    '''
    For battery swap or unlimited range.
    '''
    def __init__(self, v_index, temp_db, range_parameter):

        [setattr(self, k, param_interpret(v)) for k, v in range_parameter.items()]
        
        self.range_dist = RestrValueObject('range', v_index, temp_db, self.max_range,0,self.init_value,self.signal_list)

    def travel_step(self,distance):
        distance = self.range_dist.subtract_value()
        return distance

    def charge_travelling(self):
        self.range_dist.set_to_max()



# Travel
# ----------------------------------------------------------------------------------------------------------------


def travel_parameter(
        travel_type = 'street',
        speed       = 1,
        ):
    return {
            'travel_type': travel_type,
            'speed': speed,
            }


class StreetTravel:
    '''
    Street travel only allows horizontal and vertical movement to simulate a street grid. Used by ground vehicles.
    '''
    def __init__(self, range_obj, speed=1):
        self.range_obj = range_obj
        self.speed     = speed
        self.type      = 'street'

    def travel(self,direction):
        distance = np.sum(np.abs(direction))
        distance = self.range_obj.travel_step(distance)
        return distance*self.speed


class ArialTravel:
    '''
    Arial travel is measeured by euclidian distance. Used by drones.
    '''
    def __init__(self, range_obj, speed=1):
        self.range_obj = range_obj
        self.speed     = speed
        self.type      = 'arial'

    def travel(self,direction):
        distance = np.linalg.norm(direction)
        distance = self.range_obj.travel_step(distance)
        return distance*self.speed


# Cargo
# ----------------------------------------------------------------------------------------------------------------


def cargo_parameter(
        cargo_type         = 'standard+extra',
        max_cargo          = 10,
        max_cargo_UV       = 1,
        cargo_per_step     = 1,
        cargo_UV_per_step  = 1,
        init_value         = 0,
        signal_list        = [1,1,-1],
       ):
    return {
            'cargo_type': cargo_type,
            'max_cargo': max_cargo,
            'max_cargo_UV': max_cargo_UV,
            'cargo_per_step': cargo_per_step,
            'cargo_UV_per_step': cargo_UV_per_step,
            'init_value': init_value,
            'signal_list': signal_list,
            }


class CargoClass:
    '''
    Has normal cargo space for standard customer products and extra cargo space for UV. 
    '''
    def __init__(self, v_index, temp_db, cargo_param):

        [setattr(self, k, param_interpret(v)) for k, v in cargo_param.items()]

        self.cargo_per_step    = RestrValueObject('cargo_rate', v_index, temp_db, self.cargo_per_step, 0, self.cargo_per_step, self.signal_list)
        self.cargo_UV_per_step = RestrValueObject('cargo_UV_rate', v_index, temp_db, self.cargo_UV_per_step, 0, self.cargo_UV_per_step, self.signal_list)

        if self.cargo_type == 'standard+extra':
            self.standard_cargo = RestrValueObject('cargo', v_index, temp_db, self.max_cargo, 0, self.init_value, self.signal_list)
            self.UV_cargo       = RestrValueObject('cargo_v', v_index, temp_db, self.max_cargo_UV, 0, self.init_value, self.signal_list)
            
        elif self.cargo_type == 'standard+including':
            self.standard_cargo = RestrValueObject('cargo', v_index, temp_db, self.max_cargo, 0, self.init_value, self.signal_list)
            self.UV_cargo       = self.standard_cargo

        elif self.cargo_type == 'standard':
            self.standard_cargo = RestrValueObject('cargo', v_index, temp_db, self.max_cargo, 0, self.init_value, sself.ignal_list)

        else:
            raise Exception("cargo_type was set to '{}', but has to be: 'standard+extra', 'standard+including', 'only_standard'".format(self.cargo_type))



# Base Vehicle Classes
# ----------------------------------------------------------------------------------------------------------------

class VehicleClass:
    '''
    Creates a vehicle based on cargo_obj and travel_obj.
    Cargo loading/ unloading is restricted either by possibles actions or automatic management through heuristics.
    Travel is based on taking a direction.
    '''
    def __init__(self, temp_db, v_index, v_type, v_loadable, cargo_obj, travel_obj, weight=0):

        self.temp_db = temp_db
        self.v_index = v_index
        self.v_type = v_type
        self.v_loadable = v_loadable

        self.weight = weight
        
        self.cargo_obj  = cargo_obj
        self.travel_obj = travel_obj

        self.cur_destination = None

    def update_destination(self, coord):

        self.cur_destination = coord
        _, time = self.calc_travel()
        self.temp_db.time_till_destination[self.v_index] = time + 1

    def calc_travel(self, step_time=1):
        direction        = self.cur_destination-self.cur_coordinates
        abs_traveled     = self.travel_obj.travel(direction)
        new_coordinates  = (direction/direction)*abs_traveled*step_time
            
        time_till_destination = sum(abs(self.cur_destination - new_coordinates)) * ((abs_traveled * step_time) / sum(abs(direction)))
        return new_coordinates, time_till_destination

    def travel_period(self, step_time):
        '''
        Travel from current coordinates in direction of passed coordinates based on initialized speed.
        '''
        if self.cur_destination is not None and self.cur_destination != self.cur_coordinates:

            self.cur_coordinates, self.temp_db.time_till_destination[self.v_index] = self.calc_travel(step_time)

            self.temp_db.status_dict['v_coord'][self.v_index] = self.cur_coordinates
            
            loaded_v = self.temp_db.v_transporting_v[self.v_index]
            if any(loaded_v):
                for v in lookup_db(self.temp_db.base_groups['vehicles'], loaded_v): v.cur_coordinates = self.cur_coordinates

        if self.cur_destination is None:
            self.temp_db.time_till_destination[self.v_index] = 0


# Vehicle Creator
# ----------------------------------------------------------------------------------------------------------------

class VehicleCreator:

    def __init__(self, temp_db,
                 # Manned Vehicles:
                 MV_cargo_param, 
                 MV_range_param, 
                 MV_travel_param,
                 # Unmanned Vehicles:
                 UV_cargo_param, 
                 UV_range_param, 
                 UV_travel_param,
                 ):

        self.temp_db         = temp_db
        self.nodes_obj       = nodes_obj

        self.MV_cargo_param  = MV_cargo_param
        self.MV_range_param  = MV_range_param
        self.MV_travel_param = MV_travel_param

        self.UV_cargo_param  = UV_cargo_param
        self.UV_range_param  = UV_range_param
        self.UV_travel_param = UV_travel_param


    def create_vehicles(self, num_transporter=2, num_loadable=4, weight_loadable=1):

        #### LOADABLE ergänzen######
    
        #MV_list        = []
        #UV_per_MV_list = []

        v_index = 0

        for i in range(num_transporter):
            self.create_vehicle(v_index, True, False, self.MV_cargo_param, self.MV_range_param, self.MV_travel_param)
            v_index += 1

        for i in range(num_loadable):
            vehicle = self.create_vehicle(v_index, False, True, self.MV_cargo_param, self.MV_range_param, self.MV_travel_param, weight_loadable)
            v_index += 1


    def create_vehicle(self, v_index, can_load_v, loadable, cargo_param, range_param, travel_param, weight=0):


        cargo_obj = CargoClass(v_index, self.temp_db, cargo_param)

        # Initilize Range
        # Range calculation through batter:
        if range_param['range_type'] == 'battery':
            range_obj = StandardBattery(v_index, self.temp_db, range_param)
        # Range calculation with max distance or no range restriction if max_range = None:
        elif range_param['range_type'] == 'range':
            range_obj = StandardRange(v_index, self.temp_db, range_param)
        # Exception:
        else:
            raise Exception("range_type was set to '{}', but has to be: 'battery', 'range'".format(range_type))


        # Initilize Travel
        # Travel by street:
        if travel_param['travel_type'] == 'street':
            travel_obj = StreetTravel(v_index, self.temp_db, range_obj, travel_param)
        # Travel by air:
        elif travel_param['travel_type'] == 'arial':
            travel_obj = ArialTravel(v_index, self.temp_db, range_obj, travel_param)
        # Exception:
        else:
            raise Exception("travel_type was set to '{}', but has to be: 'street', 'arial'".format(travel_type))

        vehicle = VehicleClass(self.temp_db, v_index, can_load_v, loadable, cargo_obj, travel_obj, weight)
        self.temp_db.add_vehicle(
            vehicle, travel_param['travel_type'], range_param['range_type'], travel_obj.speed)

