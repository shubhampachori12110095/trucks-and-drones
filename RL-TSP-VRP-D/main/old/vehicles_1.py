import numpy as np
from restrictions import RestrValueObject


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
    def __init__(self,vehicle_name,logger,max_charge,charge_per_step,discharge_per_step,init_value=0,signal_list=[1,1,-1]):
        self.SoC = RestrValueObject(vehicle_name+'_SoC',logger, max_restr=max_charge,min_restr=0,init_value=0,signal_list=[1,1,-1])
        self.charge_per_step    = charge_per_step
        self.discharge_per_step = discharge_per_step

    def travel_step(self,distance):
        real_discharge = self.SoC.subtract_value(distance*self.discharge_per_step)
        distance = real_discharge/self.discharge_per_step
        return distance

    def charge_travelling(self):
        real_charge = self.SoC.add_value(self.charge_per_step)


def range_parameter(range_type       = 'range',
                    max_range        = None,
                    init_value_type  = 0,
                    signal_list      = [1,1-1],
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
    def __init__(self,vehicle_name,logger,max_range,init_value=0,signal_list=[1,1,-1])
        self.range_dist = RestrValueObject(vehicle_name+'_range',logger, max_restr=max_range,min_restr=0,init_value=0,signal_list=[1,1,-1])

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


class StandardAndExtraCargoForUV:
    '''
    Has normal cargo space for standard customer products and extra cargo space for UV. 
    '''
    def __init__(self, vehicle_name, logger,
                 # Standard Cargo:
                 max_cargo=10, cargo_per_step=1,
                 # UV Cargo:
                 max_cargo_UV=1, cargo_UV_per_step=1,
                 # Init:
                 init_value=0, signal_list=[1,1,-1]):

        # Init Standard Cargo:
        self.standard_cargo = RestrValueObject(vehicle_name+'_standard_cargo', logger, max_cargo, 0, init_value, signal_list)
        self.cargo_per_step = cargo_per_step

        # Init Standard Cargo to store UV:
        self.UV_cargo          = RestrValueObject(vehicle_name+'_UV_cargo', logger, max_cargo_UV, 0, init_value, signal_list)
        self.cargo_UV_per_step = cargo_UV_per_step

    def load_cargo(self,amount_pct):
        self.standard_cargo.add_value(amount_pct*self.cargo_per_step)

    def unload_cargo(self,amount_pct):
        self.standard_cargo.subtract_value(amount_pct*self.cargo_per_step)

    def load_UV_cargo(self,amount_pct):
        self.UV_cargo.add_value(amount_pct*self.cargo_UV_per_step)

    def unload_UV_cargo(self,amount_pct):
        self.UV_cargo.subtract_value(amount_pct*self.cargo_UV_per_step)


class StandardCargoWithUV:
    '''
    Has cargo space that includes standard customer products and UV. 
    '''
    def __init__(self, vehicle_name, logger,
                 # Cargo with UV inclusive:
                 max_cargo=10, cargo_per_step=1, cargo_UV_per_step=1,
                 # init:
                 init_value=0, signal_list=[1,1,-1]):

        self.all_cargo         = RestrValueObject(vehicle_name+'_cargo_standard_and_UV', logger, max_cargo, 0, init_value, signal_list)
        self.cargo_per_step    = cargo_per_step
        self.cargo_UV_per_step = cargo_UV_per_step

    def load_cargo(self,amount_pct):
        self.all_cargo.add_value(amount_pct*self.cargo_per_step)

    def unload_cargo(self,amount_pct):
        self.all_cargo.subtract_value(amount_pct*self.cargo_per_step)

    def load_UV_cargo(self,amount_pct):
        self.all_cargo.add_value(amount_pct*self.cargo_UV_per_step)

    def unload_UV_cargo(self,amount_pct):
        self.all_cargo.subtract_value(amount_pct*self.cargo_UV_per_step)


class StandardOnlyCargo:
    '''
    Only has cargo space for standard customer products. 
    '''
    def __init__(self, vehicle_name, logger,
                 # Only standard cargo:
                 max_cargo=10,cargo_per_step=1,
                 # Init:
                 init_value=0, signal_list=[1,1,-1]):

        self.standard_cargo    = RestrValueObject(vehicle_name+'_cargo_only_standard', logger, max_cargo, 0, init_value, signal_list)
        self.cargo_per_step    = cargo_per_step

    def load_cargo(self,amount_pct):
        self.standard_cargo.add_value(amount_pct*self.cargo_per_step)

    def unload_cargo(self,amount_pct):
        self.standard_cargo.subtract_value(amount_pct*self.cargo_per_step)



# Coordinates
# ----------------------------------------------------------------------------------------------------------------


class ExactCoordinates:
    '''
    Uses the exact coordinates chosen by the agent.
    '''
    def __init__(self,nodes_obj):
        self.nodes_obj = nodes_obj

    def transform(self, coordinates):
        return coordinates


class AutoNodeChooser:
    '''
    Alters the coordinates to the nearest customer (or depot) coordinates.
    '''
    def __init__(self,nodes_obj):
        self.nodes_obj = nodes_obj

    def transform(self, coordinates):
        nodes = self.nodes_obj.__dict__['curr_nodes']
        dist_2 = np.sum((nodes - coordinates)**2, axis=1)
        coordinates = nodes[np.argmin(dist_2)]
        return transformed_coordinates

    def set_new_placement(self, distance, direction):
        self.coordinator.check_and_move(distance, direction)


class DiscreteNodeChooser:
    '''
    Chooses the coordinates based on a discrete action, which will be a customer (or a depot).
    '''
    def __init__(self,nodes_obj):
        self.nodes_obj = nodes_obj

    def transform(self, discrete_node):
        nodes = self.nodes_obj.__dict__['curr_nodes']
        coordinates = nodes[discrete_node[0]]
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


        self.logger       = self.vehicle_obj.__dict__['cargo_obj'].__dict__['logger']
        self.vehicle_name = self.vehicle_obj.__dict__['cargo_obj'].__dict__['vehicle_name']
        self.travel_type  = self.vehicle_obj.__dict__['travel_obj'].__dict__['travel_type']

        self.non_viol_signal  = signal_list[0]
        self.viol_signal      = signal_list[2]

        self.visualize = visualize

    def set_coordinates(coordinates):
        '''
        Init some starting coordinates eg. the starting depot.
        '''
        self.cur_coordinates = coordinates

    def travel_to(self,coordinates):
        '''
        Travel from current coordinates in direction of passed coordinates based on initialized speed.
        '''
        # Transform coordinates based on coord_obj
        coordinates = self.coord_obj.transform(coordinates)

        # Calculate coordinates that can be travelled in one step.
        # New coordinates will be in the direction of the coordinates by the agent.
        direction       = coordinates-self.cur_coordinates
        abs_traveled    = self.travel_obj.travel(direction)
        new_coordinates = (direction/direction)*abs_traveled
        
        # Draw traveled route:
        if visualize == True:
            self.logger.coordinates(self.vehicle_name, self.travel_type, new_coordinates)
        return new_coordinates



    def load_cargo(self,amount_pct):
        '''
        Load standard cargo, can only be used at customer node.
        '''
        self.cargo_obj.load_cargo(amount_pct)

    def unload_cargo(self,amount_pct):
        '''
        Unload standard cargo, can only be used at customer node.
        '''
        self.cargo_obj.unload_cargo(amount_pct)

    def load_UV_cargo(self,amount_pct):
        '''
        Load UV to cargo, can only be used when meeting a UV.
        '''
        self.cargo_obj.load_UV_cargo(amount_pct)

    def unload_UV_cargo(self,amount_pct):
        '''
        Unload UV from cargo.
        '''
        self.cargo_obj.unload_UV_cargo(amount_pct)


class LockedTravelVehicleClass(VehicleClass):

    def __init__(self, vehicle_name, logger, cargo_obj, travel_obj, coord_obj, visualize, signal_list):
        super().__init__(vehicle_name, logger, cargo_obj, travel_obj, coord_obj, visualize, signal_list)
        
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
        self.logger.coordinates_signal(self.vehicle_name, self.travel_type, restr_signal)

        # Draw traveled route:
        if self.visualize == True:
            self.logger.coordinates(self.vehicle_name, self.travel_type, new_coordinates)
        
        return new_coordinates



# Vehicle Creator
# ----------------------------------------------------------------------------------------------------------------


class VehicleCreator:

    def __init__(self, logger, nodes_obj,
                 coord_type='exact', locked_travel=False, visualize=False, signal_list=[1,1-1],
                 # Manned Vehicles:
                 MV_cargo_param  = MV_cargo_parameter(), 
                 MV_range_param  = range_parameter(), 
                 MV_travel_param = travel_parameter(),
                 # Unmanned Vehicles:
                 UV_cargo_param  = UV_cargo_parameter(), 
                 UV_range_param  = battery_parameter(), 
                 UV_travel_param = travel_parameter(travel_type='arial'),
                 ):

        self.logger          = logger
        self.nodes_obj       = nodes_obj

        self.coord_type      = coord_type
        self.locked_travel   = locked_travel
        self.visualize       = visualize
        self.signal_list     = signal_list

        self.MV_cargo_param  = MV_cargo_param
        self.MV_range_param  = MV_range_param
        self.MV_travel_param = MV_travel_param

        self.UV_cargo_param  = UV_cargo_param
        self.UV_range_param  = UV_range_param
        self.UV_travel_param = UV_travel_param


    def create_vehicles(self, num_MV=2,num_UV_per_MV=2):
    
        MV_list        = []
        UV_per_MV_list = []
        
        for i in range(num_MV):
            MV_list.append(self.create_vehicle('MV_street_'+str(i), self.MV_cargo_param, self.MV_range_param, self.MV_travel_param))
            UV_list = []
            for j in range(num_UV_per_MV):
                UV_list.append(self.create_vehicle('UV_arial_'+str(i)+'_'+str(j), self.UV_cargo_param, self.UV_range_param, self.UV_travel_param))
            UV_per_MV_list.append(UV_list)

        return MV_list, UV_per_MV_list


    def create_vehicle(self, vehicle_name, cargo_param, range_param, travel_param):

        # Initilize Cargo
        # Standard Cargo with extra cargo for UV:
        if cargo_param['cargo_type'] == 'standard+extra':
            cargo_obj = StandardAndExtraCargoForUV(vehicle_name, self.logger, cargo_param)
        # Standard cargo that includes UV:
        elif cargo_param['cargo_type'] == 'standard+including':
            cargo_obj = StandardCargoWithUV(vehicle_name, self.logger, cargo_param)
        # Standard cargo without UV:
        elif cargo_param['cargo_type'] == 'standard':
            cargo_obj = StandardOnlyCargo(vehicle_name, self.logger, cargo_param)
        # Exception:
        else:
            raise Exception("cargo_type was set to '{}', but has to be: 'standard+extra', 'standard+including', 'only_standard'".format(cargo_type))


        # Initilize Range
        # Range calculation through batter:
        if range_param['range_type'] == 'battery':
            range_obj = StandardBattery(vehicle_name, self.logger, range_param)
        # Range calculation with max distance or no range restriction if max_range = None:
        elif range_param['range_type'] == 'range':
            range_obj = StandardRange(vehicle_name, self.logger, range_param)
        # Exception:
        else:
            raise Exception("range_type was set to '{}', but has to be: 'battery', 'range'".format(range_type))


        # Initilize Travel
        # Travel by street:
        if travel_param['travel_type'] == 'street':
            travel_obj = StreetTravel(vehicle_name, self.logger, range_obj, travel_param)
        # Travel by air:
        elif travel_param['travel_type'] == 'arial':
            travel_obj = ArialTravel(vehicle_name, self.logger, range_obj, travel_param)
        # Exception:
        else:
            raise Exception("travel_type was set to '{}', but has to be: 'street', 'arial'".format(travel_type))


        # Initilize Coordinate Chooser
        # Always travel to exact coordinates:
        if coord_type == 'exact':
            coord_obj = ExactCoordinates(self.nodes_obj)
        # Travel to nearest node or depot:
        elif coord_type == 'auto':
            coord_obj = AutoNodeChooser(self.nodes_obj)
        # Travel to depot, used for disrcete action:
        elif coord_type == 'discrete':
            coord_obj = DiscreteNodeChooser(self.nodes_obj)
        # Exception:
        else:
            raise Exception("coord_type was set to '{}', but has to be: 'exact', 'auto', 'discrete'".format(coord_type))

        # Check if coordinates should be locked till vehicle reaches the destination:
        if locked_travel:
            return LockedTravelVehicleClass(cargo_obj, travel_obj, coord_obj, self.visualize, self.signal_list)
        
        # No coordinates locking:
        return VehicleClass(cargo_obj, travel_obj, coord_obj, self.visualize, self.signal_list)
