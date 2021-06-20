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


'''
def v_load_v(transporter_obj, to_load_obj_list, cargo_amount_list):
    '''
    '''

    num_v_to_load = min(l_ignore_none([transporter_obj.cargo_obj.vehicle_per_step, len(to_load_obj_list)]))

    loaded_list = []
    for i in range(num_v_to_load):

        weight = to_load_obj_list[i].weight

        cargo_amount = min(
            to_load_obj_list[i].cargo_obj.cargo_per_step.check_subtract_value(cargo_amount_list[i]),
            to_load_obj_list[i].cargo_obj.standard_cargo.check_subtract_value(cargo_amount_list[i]),
            transporter_obj.cargo_obj.cargo_per_step.check_subtract_value(cargo_amount_list[i]+weight),
            transporter_obj.cargo_obj.standard_cargo.check_add_value(cargo_amount_list[i]+weight)
            )

        if cargo_amount >= to_load_obj_list[i].cargo_obj.standard_cargo.cur_value:
            to_load_obj_list[i].cargo_obj.cargo_per_step.subtract_value(cargo_amount-weight),
            to_load_obj_list[i].cargo_obj.standard_cargo.subtract_value(cargo_amount-weight),
            transporter_obj.cargo_obj.cargo_per_step.subtract_value(cargo_amount),
            transporter_obj.cargo_obj.standard_cargo.add_value(cargo_amount)

            loaded_list.append(to_load_obj_list[i].name)

    return loaded_list
'''
####################################################################################################################


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