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
###############################################################################################################################################




    def unload_vehicles(self, v_j=None):

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


    def load_vehicle(self, v_j=None):

        if vehicle_j is None:
            

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


    def unload_cargo(self, n_j=None, amount=None):

        if customer_j is None:
            

        if self.temp_db.same_coord(vehicle_i, customer_j, 'c_coord'):
            real_amount = vehicle_at_customer(self.temp_db.base_groups['vehicles'][vehicle_i], self.temp_db.base_groups['customers'][customer_j], amount)

            print('unloaded cargo',real_amount,'from',vehicle_i,'to customer',customer_j)
            if real_amount > 0:
                self.v_did_sth = True

    def load_cargo(self, n_j=None, amount=None):

        if depot_j is None:
            
    
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

##############################################################################################################################################################################


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
