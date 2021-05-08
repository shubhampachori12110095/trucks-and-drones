import numpy as np

'''
- euclidean route
- non-euclidean route

Abbreviations:

MV: Manned Vehicle
UV: Unmanned Vehicle
SoC: State of Charge
'''

#####################################
######## Restriction Classes ########
#####################################

class MinToMaxRestriction:
    '''
    Used for standard restrictions.The signal list follows this order:
    - If no restriction is violated, its called non violation (non_viol_signal = signal_list[0])
    - If the restriction is violated, but the changed amount is greater than zero its a semi violation (semi_viol_signal = signal_list[1])
    - A restriction violation resulting in a changed amount of zero will be interpreted as a full violation (viol_signal = signal_list[2])
    The violation signal can be used to calculate a reward or penalization.
    '''
    def __init__(self, max_restr, min_restr, signal_list):
        self.max_restr = max_restr
        self.min_restr = min_restr

        self.non_viol_signal  = signal_list[0]
        self.semi_viol_signal = signal_list[1]
        self.viol_signal      = signal_list[2]


    def add_value(self, cur_value, value):
    '''
    Adds a specified amount to the current value under the initialzied max restriction.
    '''
        new_value = cur_value + new_value
        
        if new_value <= self.max_restr:
            cur_value = new_value
            # return 1 to signal NO violation
            return cur_value, self.non_viol_signal
        elif cur_value == self.max_restr:
            # return -1 to signal violation
            return cur_value, self.viol_signal
        else:
            return cur_value, self.semi_viol_signal


    def subtract_value(self, cur_value, value):
        '''
        Subtracts a specified amount from the current value under the initialzied min restriction.
        '''
        new_value = cur_value - new_value
        
        if new_value >= self.min_restr:
            cur_value = new_value
            # return 1 to signal NO violation
            return cur_value, self.non_viol_signal
        elif cur_value == self.min_restr:
            # return -1 to signal violation
            return cur_value, self.viol_signal
        else:
            return cur_value, self.semi_viol_signal


class MinRestriction(MinToMaxRestriction):
    '''
    Extension for MinToMaxRestriction class, that excludes the max restriction.
    '''
    def __init__(self, min_restr, signal_list):
        super().__init__(None, min_restr, signal_list)

    def add_value(self, cur_value, value):
        '''
        Overrides the max restriction from the original class to unrestricted max.
        '''
        cur_value += value
        return cur_value, self.non_viol_signal


class MaxRestriction(MinToMaxRestriction):
    '''
    Extension for MinToMaxRestriction class, that excludes the min restriction.
    '''
    def __init__(self, max_restr, signal_list):
        super().__init__(max_restr, None, signal_list)

    def subtract_value(self, cur_value, value):
        '''
        Overrides the min restriction from the original class to unrestricted min.
        '''
        cur_value -= value
        return cur_value, self.non_viol_signal


class DummyRestriction(MinToMaxRestriction):
    '''
    Used when no restrictions are needed. Extension for MinToMaxRestriction class, that excludes both the min and the max restriction
    '''
    def __init__(self, signal_list):
        super().__init__(None, None, signal_list)

    def add_value(self, cur_value, value):
        '''
        Overrides the max restriction from the original class to unrestricted max.
        '''
        cur_value += value
        return cur_value, self.non_viol_signal

    def subtract_value(self, cur_value, value):
        '''
        Overrides the min restriction from the original class to unrestricted min.
        '''
        cur_value -= value
        return cur_value, self.non_viol_signal


def RestrValueObject:
    '''
    Traces a the value of a variable that is restricted. Can also be used to trace unrestriced variabels, in which case a dummy restriction will be created (doesn't restrict anything).
    '''

    def __init__(self, name, logger, max_restr=None,min_restr=None,init_value=0,signal_list=[1,1,-1]):

        self.name   = name
        self.logger = logger

        self.max_restr  = max_restr
        self.init_value = init_value
        self.reset()

        if max_restr == None and min_restr == None:
            self.restriction = DummyRestriction(signal_list)        
        elif max_restr == None and min_restr != None:
            self.restriction = MinRestriction(min_restr,signal_list)        
        elif max_restr != None and min_restr == None:
            self.restriction = MaxRestriction(max_restr,signal_list)
        else:
            self.restriction = MinToMaxRestriction(max_restr,min_restr,signal_list)

    def reset(self, step=None):
        self.cur_value = self.init_value
        if step != None:
            self.logger.value(self.name,new_value,step)

    def set_to_max(self):
        self.cur_value = self.max_restr

    def update(self, new_value, restr_signal, step):
        self.logger.value(self.name,  new_value,    step)
        self.logger.signal(self.name, restr_signal, step)
        self.cur_value = new_value

    def add_value(self, value, step):
        new_value, restr_signal = self.restriction.add_value(self.cur_value,value)
        self.update(new_value,restr_signal,step)
        return new_value

    def subtract_value(self, value, step):
        new_value, restr_signal = self.restriction.subtract_value(self.cur_value,value)
        self.update(new_value,restr_signal,step)
        return new_value

    def check_max_and_go(self, value, step):
        new_value, restr_signal = self.restriction.add_value(self.cur_value,value)
        if restr_signal != self.viol_signal:
            self.update(new_value,restr_signal,step)
            return True
        else:
            self.update(self.cur_value,restr_signal,step)
            return False

    def check_min_and_go(self, value, step):
        new_value, restr_signal = self.restriction.subtract_value(self.cur_value,value)
        if restr_signal != self.viol_signal:
            self.update(new_value,restr_signal,step)
            return True
        else:
            self.update(self.cur_value,restr_signal,step)
            return False


###############################
######## Range Classes ########
###############################

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

################################
######## Travel Classes ########
################################

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

###############################
######## Cargo Classes ########
###############################

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


class StandardCargoWithUV
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


class StandardOnlyCargo
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


###############################
######## Vehicle Class ########
###############################

class VehicleClass:
    '''
    Creates a vehicle based on cargo_obj and travel_obj.
    Cargo loading/ unloading is restricted either by possibles actions or automatic management through heuristics.
    Travel is based on taking a direction.
    '''
    def __init__(self, cargo_obj, travel_obj):
        self.cargo_obj  = cargo_obj
        self.travel_obj = travel_obj

    def set_coordinates(coordinates):
        '''
        Init some starting coordinates eg. the starting depot.
        '''
        self.cur_coordinates = coordinates

    def travel_to(self,coordinates):
        '''
        Travel from current coordinates in direction of passed coordinates based on initialized speed.
        '''
        direction = coordinates-self.cur_coordinates
        abs_traveled = self.travel_obj.travel(direction)
        new_coordinates = (direction/direction)*abs_traveled
        return

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

def create_vehicle(vehicle_name, logger, travel_type, cargo_type, ...................)

#####################################
######## Coordinates Classes ########
#####################################

class CoordinatesLocking:

    def __init__(self, vehicle_obj, signal_list, visualize):
        self.vehicle_obj  = vehicle_obj
        self.logger       = self.vehicle_obj.__dict__['cargo_obj'].__dict__['logger']
        self.vehicle_name = self.vehicle_obj.__dict__['cargo_obj'].__dict__['vehicle_name']
        self.travel_type  = self.vehicle_obj.__dict__['travel_obj'].__dict__['travel_type']

        self.non_viol_signal  = signal_list[0]
        self.viol_signal      = signal_list[2]

        self.visualize = visualize

        self.reached = True

    def go_to(self,coordinates):

        if self.reached == True:
            self.go_to_coordinates = coordinates
            restr_signal = self.non_viol_signal
        else:
            if self.go_to_coordinates  != coordinates:
                restr_signal = self.viol_signal
        
        new_coordinates = self.vehicle_obj.travel_to(self.go_to_coordinates)

        if new_coordinates == self.go_to_coordinates:
            self.reached = True
        else:
            self.rechead = False

        self.logger.coordinates_signal(self.vehicle_name, self.travel_type, restr_signal)

        if visualize == True:
            self.logger.coordinates(self.vehicle_name, self.travel_type, new_coordinates)

        return new_coordinates



class CoordinatesNotLocked:

    def __init__(self, vehicle_obj, visualize):
        self.vehicle_obj  = vehicle_obj
        self.logger       = self.vehicle_obj.__dict__['cargo_obj'].__dict__['logger']
        self.vehicle_name = self.vehicle_obj.__dict__['cargo_obj'].__dict__['vehicle_name']
        self.travel_type  = self.vehicle_obj.__dict__['travel_obj'].__dict__['travel_type']

        self.visualize = visualize

    def go_to(self,coordinates):

        new_coordinates = self.vehicle_obj.travel_to(self.go_to_coordinates)
        if visualize == True:
            self.logger.coordinates(self.vehicle_name, self.travel_type, new_coordinates)
        return new_coordinates


class ExactCoordinates:
    '''
    Uses the exact coordinates chosen by the agent.
    '''
    def __init__(self,vehicle_obj, nodes_obj, signal_list=[1,1,-1], lock_coordinates=False, visualize=False):
        if lock_coordinates == False:
            self.coordinator = CoordinatesNotLocked(vehicle_obj,visualize)
        else:
            self.coordinator = CoordinatesLocking(vehicle_obj, signal_list,visualize)

    def travel_in_direction(coordinates):
        return self.coordinator.go_to(coordinates)


class AutoNodeChooser:
    '''
    Alters the coordinates to the nearest customer (or depot) coordinates.
    '''
    def __init__(self,vehicle_obj, nodes_obj, signal_list=[1,1,-1], lock_coordinates=False):
        self.nodes_obj = nodes_obj

        if lock_coordinates == False:
            self.coordinator = CoordinatesNotLocked(vehicle_obj,visualize)
        else:
            self.coordinator = CoordinatesLocking(vehicle_obj, signal_list,visualize)

    def travel_in_direction(coordinates):
        nodes = self.nodes_obj.__dict__['curr_nodes']
        dist_2 = np.sum((nodes - coordinates)**2, axis=1)
        coordinates = nodes[np.argmin(dist_2)]
        return self.coordinator.go_to(coordinates)


class DiscreteNodeChooser:
    '''
    Chooses the coordinates based on a discrete action, which will be a customer (or a depot).
    '''
    def __init__(self,vehicle_obj, nodes_obj, signal_list=[1,1,-1], lock_coordinates=False):
        self.nodes_obj = nodes_obj

        if lock_coordinates == False:
            self.coordinator = CoordinatesNotLocked(vehicle_obj,visualize)
        else:
            self.coordinator = CoordinatesLocking(vehicle_obj, signal_list,visualize)

    def travel_in_direction(discrete_node):
        nodes = self.nodes_obj.__dict__['curr_nodes']
        coordinates = nodes[discrete_node[0]]
        return self.coordinator.go_to(coordinates)






class StaticCustomers:

    def __init__(self,num_customers=10):

        self.num_customers = num_customers

    def reset_grid(self):

class DynamicCustomers:

    def __init__(self,num_customers=10):

        self.num_customers = num_customers


class AutoCustomerChooser:


class ExactCustomerChooser:


class DiscreteCustomerChooser:


class StandardGrid:

    def __init__(self, MV=GroundVehicle(), UV=ArialVehicle(), x_size=100, y_size=100, num_MV=1, num_UV_per_MV=1):

        self.MV = MV
        self.UV = UV

        self.x_size = x_size
        self.y_size = y_size

        self.num_customers = num_customers
        self.num_MV        = num_MV
        self.num_UV_per_MV = num_UV_per_MV

    def 

