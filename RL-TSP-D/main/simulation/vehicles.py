'''

'''

import numpy as np
from restrictions import RestrValueObject
from common_sim_func import param_interpret, random_coordinates


# Range
# ----------------------------------------------------------------------------------------------------------------


def battery_parameter(range_type         = 'battery',
                      max_charge         = 100,
                      charge_per_step    = 50,
                      discharge_per_step = 10,
                      init_value         = 0,
                      signal_list        = [1,1-1],
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
    def __init__(self,vehicle_name,temp_db,battery_parameter):

        [setattr(self, k, param_interpret(v)) for k, v in battery_parameter.items()]
        
        self.SoC = RestrValueObject(self.max_charge, 0, self.init_value, self.signal_list)
        temp_db.add_restriction(self.range_dist, vehicle_name+'_SoC', base_group='battery')


    def travel_step(self,distance):
        real_discharge = self.SoC.subtract_value(distance*self.discharge_per_step)
        distance = real_discharge/self.discharge_per_step
        return distance

    def charge_travelling(self):
        real_charge = self.SoC.add_value(self.charge_per_step)


def range_parameter(range_type  = 'range',
                    max_range   = None,
                    init_value  = 0,
                    signal_list = [1,1-1],
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
    def __init__(self, vehicle_name, temp_db, range_parameter):

        [setattr(self, k, param_interpret(v)) for k, v in range_parameter.items()]
        
        self.range_dist = RestrValueObject(self.max_range,0,self.init_value,self.signal_list)
        temp_db.add_restriction(self.range_dist, vehicle_name+'_range', base_group='range')

    def travel_step(self,distance):
        distance = self.range_dist.subtract_value()
        return distance

    def charge_travelling(self):
        self.range_dist.set_to_max()



# Travel
# ----------------------------------------------------------------------------------------------------------------


def travel_parameter(travel_type = 'street',
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


def MV_cargo_parameter(cargo_type         = 'standard+extra',
                       max_cargo          = 10,
                       max_cargo_UV       = 1,
                       cargo_weigth_UV    = 1,
                       cargo_per_step     = 1,
                       cargo_UV_per_step  = 1,
                       init_value         = 0,
                       signal_list        = [1,1-1],
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


def UV_cargo_parameter(cargo_type         = 'standard',
                       max_cargo          = 1,
                       max_cargo_UV       = 0,
                       cargo_per_step     = 1,
                       cargo_UV_per_step  = 0,
                       init_value         = 0,
                       signal_list        = [1,1-1],
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
    def __init__(self, vehicle_name, temp_db, cargo_param):

        self.name = vehicle_name

        [setattr(self, k, param_interpret(v)) for k, v in cargo_param.items()]

        self.cargo_per_step    = RestrValueObject(self.cargo_per_step, 0, self.cargo_per_step, self.signal_list)
        self.cargo_UV_per_step = RestrValueObject(self.cargo_UV_per_step, 0, self.cargo_UV_per_step, self.signal_list)

        temp_db.add_restriction(self.cargo_per_step, vehicle_name+'_cargo_per_step', base_group='cargo_rate')
        temp_db.add_restriction(self.cargo_UV_per_step, vehicle_name+'_cargo_UV_per_step', base_group='cargo_UV_rate')

        if self.cargo_type == 'standard+extra':
            self.standard_cargo = RestrValueObject(self.max_cargo, 0, self.init_value, self.signal_list)
            self.UV_cargo       = RestrValueObject(self.max_cargo_UV, 0, self.init_value, self.signal_list)
            
            temp_db.add_restriction(self.standard_cargo, vehicle_name+'_standard_cargo', base_group='cargo')
            temp_db.add_restriction(self.cargo_UV_per_step, vehicle_name+'_UV_cargo', base_group='cargo_UV')

        elif self.cargo_type == 'standard+including':
            self.standard_cargo = RestrValueObject(self.max_cargo, 0, self.init_value, self.signal_list)
            self.UV_cargo       = self.standard_cargo

            temp_db.add_restriction(self.standard_cargo, vehicle_name+'_standard_cargo', base_group='cargo')

        elif self.cargo_type == 'standard':
            self.standard_cargo = RestrValueObject(self.max_cargo, 0, self.init_value, sself.ignal_list)
            temp_db.add_restriction(self.standard_cargo, vehicle_name+'_standard_cargo', base_group='cargo')

        else:
            raise Exception("cargo_type was set to '{}', but has to be: 'standard+extra', 'standard+including', 'only_standard'".format(self.cargo_type))



# Coordinates ####################### MOVE TO ACTION INTERPRETER #######################
# ----------------------------------------------------------------------------------------------------------------


class ExactCoordinates:
    '''
    Uses the exact coordinates chosen by the agent.
    '''
    def __init__(self, temp_db):
        self.temp_db = temp_db

    def transform(self, coordinates):
        return coordinates


class AutoNodeChooser:
    '''
    Alters the coordinates to the nearest customer (or depot) coordinates.
    '''
    def __init__(self, temp_db):
        self.temp_db = temp_db

    def transform(self, coordinates):
        nodes = self.temp_db.current_nodes
        dist_2 = np.sum((nodes - coordinates)**2, axis=1)
        coordinates = nodes[np.argmin(dist_2)]
        return transformed_coordinates


class DiscreteNodeChooser:
    '''
    Chooses the coordinates based on a discrete action, which will be a customer (or a depot).
    '''
    def __init__(self, temp_db):
        self.temp_db = temp_db

    def transform(self, discrete_node):
        nodes = self.temp_db.current_nodes
        coordinates = nodes[discrete_node]
        return transformed_coordinates



# Base Vehicle Classes
# ----------------------------------------------------------------------------------------------------------------

class VehicleClass:
    '''
    Creates a vehicle based on cargo_obj and travel_obj.
    Cargo loading/ unloading is restricted either by possibles actions or automatic management through heuristics.
    Travel is based on taking a direction.
    '''
    def __init__(self, cargo_obj, travel_obj, coord_obj):

        self.cargo_obj  = cargo_obj
        self.travel_obj = travel_obj
        self.coord_obj  = coord_obj
        
    def set_coordinates(coordinates):
        '''
        Init some starting coordinates eg. the starting depot. Also used for UV transportation.
        '''
        self.cur_coordinates = coordinates

    def travel_to(self,destination):
        '''
        Travel from current coordinates in direction of passed coordinates based on initialized speed.
        '''
        # Transform coordinates based on coord_obj
        coordinates = self.coord_obj.transform(destination)

        # Calculate coordinates that can be travelled in one step.
        # New coordinates will be in the direction of the coordinates by the agent.
        direction       = coordinates-self.cur_coordinates
        abs_traveled    = self.travel_obj.travel(direction)
        new_coordinates = (direction/direction)*abs_traveled
        
        self.cur_coordinates = new_coordinates
        return new_coordinates


class LockedTravelVehicleClass(VehicleClass): #################################################################!!!!!!!

    def __init__(self, cargo_obj, travel_obj, coord_obj):
        super().__init__(argo_obj, travel_obj, coord_obj)
        
        self.reached = True

    def travel_to(self,coordinates):
        '''
        Travel from current coordinates in direction of passed coordinates based on initialized speed.
        '''

        # Transform coordinates based on coord_obj
        coordinates = self.coord_obj.transform(coordinates)

        # If destination is already rechead, new coordinates will be locked:
        if self.reached == True:
            self.go_to_coordinates = coordinates
            restr_signal = self.non_viol_signal
        # Check if new_coordinate are different to the locked ones.
        # Can be used to penalize agent if coordinates are different.
        else:
            if self.go_to_coordinates  != coordinates:
                restr_signal = self.viol_signal
            else:
                restr_signal = self.non_viol_signal

        # Calculate coordinates that can be travelled in one step.
        # New coordinates will be in the direction of the coordinates by the agent.
        direction       = coordinates-self.cur_coordinates
        abs_traveled    = self.travel_obj.travel(direction)
        new_coordinates = (direction/direction)*abs_traveled

        # Signal that new coordinates can be locked, if the vehicle reached its destination:
        if new_coordinates == self.go_to_coordinates:
            self.reached = True
        else:
            self.rechead = False

        # Store restr_signal for possible penalization
        self.temp_db.coordinates_signal(self.vehicle_name, self.travel_type, restr_signal)

        # Draw traveled route:
        if self.visualize == True:
            self.temp_db.coordinates(self.vehicle_name, self.travel_type, new_coordinates)
        
        return new_coordinates



# Vehicle Creator
# ----------------------------------------------------------------------------------------------------------------

class VehicleCreator:

    def __init__(self, temp_db,
                 coord_type='exact', locked_travel=False,
                 # Manned Vehicles:
                 MV_cargo_param  = MV_cargo_parameter(), 
                 MV_range_param  = range_parameter(), 
                 MV_travel_param = travel_parameter(),
                 # Unmanned Vehicles:
                 UV_cargo_param  = UV_cargo_parameter(), 
                 UV_range_param  = battery_parameter(), 
                 UV_travel_param = travel_parameter(travel_type='arial'),
                 ):

        self.temp_db         = temp_db
        self.nodes_obj       = nodes_obj

        self.coord_type      = coord_type
        self.locked_travel   = locked_travel

        self.MV_cargo_param  = MV_cargo_param
        self.MV_range_param  = MV_range_param
        self.MV_travel_param = MV_travel_param

        self.UV_cargo_param  = UV_cargo_param
        self.UV_range_param  = UV_range_param
        self.UV_travel_param = UV_travel_param


    def create_vehicles(self, num_MV=2,num_UV_per_MV=2):
    
        #MV_list        = []
        #UV_per_MV_list = []
        
        UV_index = 0
        for i in range(num_MV):
            MV_obj = self.create_vehicle('MV_'+str(i), self.MV_cargo_param, self.MV_range_param, self.MV_travel_param)
            #MV_list.append(MV_obj)
            self.temp_db.add_vehicle(MV_obj,'MV_'+str(i),group_list=['MV'])

            
            UV_list = []
            for j in range(num_UV_per_MV):
                UV_obj = self.create_vehicle('UV_'+str(UV_index), self.UV_cargo_param, self.UV_range_param, self.UV_travel_param)
                self.temp_db.add_vehicle(UV_obj, 'UV_'+str(UV_index), group_list=['UV'])
                UV_list.append('UV_'+str(UV_index))

            self.temp_db.MV_transporting_UV['MV_'+str(i)] = UV_list
            
            #UV_per_MV_list.append(UV_list)

        #return MV_list, UV_per_MV_list


    def create_vehicle(self, vehicle_name, cargo_param, range_param, travel_param):

        self.vehicle_name = vehicle_name
        cargo_obj = CargoClass(vehicle_name, self.temp_db, cargo_param)



        # Initilize Range
        # Range calculation through batter:
        if range_param['range_type'] == 'battery':
            range_obj = Standartemp_dbattery(vehicle_name, self.temp_db, range_param)
        # Range calculation with max distance or no range restriction if max_range = None:
        elif range_param['range_type'] == 'range':
            range_obj = StandardRange(vehicle_name, self.temp_db, range_param)
        # Exception:
        else:
            raise Exception("range_type was set to '{}', but has to be: 'battery', 'range'".format(range_type))


        # Initilize Travel
        # Travel by street:
        if travel_param['travel_type'] == 'street':
            travel_obj = StreetTravel(vehicle_name, self.temp_db, range_obj, travel_param)
        # Travel by air:
        elif travel_param['travel_type'] == 'arial':
            travel_obj = ArialTravel(vehicle_name, self.temp_db, range_obj, travel_param)
        # Exception:
        else:
            raise Exception("travel_type was set to '{}', but has to be: 'street', 'arial'".format(travel_type))


        # Initilize Coordinate Chooser
        # Always travel to exact coordinates:
        if self.coord_type == 'exact':
            coord_obj = ExactCoordinates(self.temp_db)
        # Travel to nearest node or depot:
        elif self.coord_type == 'auto':
            coord_obj = AutoNodeChooser(self.temp_db)
        # Travel to depot, used for disrcete action:
        elif self.coord_type == 'discrete':
            coord_obj = DiscreteNodeChooser(self.temp_db)
        # Exception:
        else:
            raise Exception("coord_type was set to '{}', but has to be: 'exact', 'auto', 'discrete'".format(self.coord_type))

        # Check if coordinates should be locked till vehicle reaches the destination:
        if self.locked_travel:
            return LockedTravelVehicleClass(cargo_obj, travel_obj, coord_obj)
        
        # No coordinates locking:
        return VehicleClass(cargo_obj, travel_obj, coord_obj)
