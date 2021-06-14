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



#########################################################################################################

class BaseCustomerDemand:

    def __init__(self, customer_name, temp_db, demand):

        self.customer_name = customer_name
        self.temp_db        = temp_db
        self.demand        = demand

    def step(self, day_step):
        self.temp_db.update_node(self.customer_name,self.demand.value)


class RechargingCustomerDemand():

    def __init__(self, customer_name, temp_db, demand, demand_after_steps, demand_add):

        self.customer_name = customer_name
        self.temp_db        = temp_db
        self.demand        = demand

        self.demand_after_steps = demand_after_steps
        self.demand_add         = demand_add
        self.step_count         = 0

    def step(self, day_step, count=True):
        if count:
            if self.step_count == self.demand_after_steps:
                self.demand.add_value(self.demand_add)
                self.step_count = 0
            else:
                self.step_count += 1
        self.temp_db.update_node(self.customer_name,self.demand.value)


class BaseCustomerClass:

    def __init__(self, customer_name, temp_db, customer_param):

        self.customer_name = customer_name
        self.temp_db = temp_db

        self.demand = RestrValueObject(customer_name, temp_db, max_restr=self.init_demand, min_restr=0, init_value=self.init_demand, signal_list=self.signal_list)

        if self.demand_after_steps == None:
            self.demand_base_obj = BaseCustomerDemand(customer_name, temp_db, self.demand)
        else:
            self.demand_base_obj = RechargingCustomerDemand(customer_name, temp_db, self.demand, self.demand_after_steps, self.demand_add)

    def unload_vehicle(self,amount):
        return self.demand.subtract_value(amount)

    def step(self, day_step):
        self.demand_base_obj.step(day_step)
        


class DynamicCustomerClass(BaseCustomerClass):

    def __init__(self, customer_name, temp_db, customer_param):
        super().__init__(customer_name, temp_db, customer_param)

        self.step_counter = 0
    
    def step(self, day_step):

        if day_step < self.first_demand_step:
            self.demand.zero_demand()
            self.demand_base_obj.step(day_step, False)
        
        elif day_step == self.first_demand_step:
            self.demand.set_to_max()
            self.demand_base_obj.step(day_step, False)
        
        else:
            self.demand_base_obj.step(day_step)
            