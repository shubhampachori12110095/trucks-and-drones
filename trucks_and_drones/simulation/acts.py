import gym
import numpy as np
from gym import spaces

class CustomAction:
    '''
    Base class for custom actions.
    '''

    def __init__(self):
        pass

    def build(self, temp_db, simulation):
        self.temp_db = temp_db
        self.simulation = simulation

    def action_space(self):
        pass

    def decode_actions(self, actions):
        pass

    def reset(self):
        pass

    def to_customer(
            self,
            customer_idx,
            vehicle,
            terminate_on_mistake=True,
            mistake_penalty=-10,
            additional_reward=10
    ):

        done = False
        reward = 0
        node_idx = customer_idx + self.temp_db.num_depots # transform customer index to node index

        # No demand, bad action:
        if self.temp_db.get_val('n_items')[node_idx] == 0:
            reward += mistake_penalty
            if terminate_on_mistake:
                done = True

        # Customer has demand:
        else:
            if not self.temp_db.status_dict['v_items'][vehicle.v_index] == 0:
                reward += additional_reward

        vehicle.set_node_as_destination(node_idx)

        # calculate move time and check if vehicle can actually move/ reach destination:
        time_frame, error_signal = vehicle.calc_move_time(check_if_dest_reachable=True)

        if error_signal:
            reward += mistake_penalty

        if terminate_on_mistake and error_signal:
            done = True

        return time_frame, reward, done

    def simple_demand_satisfaction(self, customer_idx):
        '''
        Assumes customers demand can be satisfied instantly without any restrictions,
        additional assumes infinite cargo of vehicle
        '''
        node_idx = customer_idx + self.temp_db.num_depots # transform customer index to node index
        self.temp_db.status_dict['n_items'][node_idx] = 0

    def check_demands(self):
        return bool(np.sum(self.temp_db.get_val('n_items')[self.temp_db.num_depots:]) == 0)


class TSPAction(CustomAction):

    def __init__(self, temp_db, simulation):
        super().__init__()
        self.temp_db = temp_db
        self.simulation = simulation

    def action_space(self):
        return spaces.Discrete(self.temp_db.num_customers)

    def decode_actions(self, actions):

        vehicle = self.temp_db.base_groups['vehicles'][0] # only one vehicle with index 0

        # calc the time to move and check if action is valid
        time_frame, reward, done = self.to_customer(
            customer_idx=int(actions),
            vehicle=vehicle
        )

        # if vehicle can move, update reward and log to total time (costs):
        # (under standard parameter/assumptions, this will always be the case)
        if not np.isnan(time_frame):
            reward -= time_frame
            self.temp_db.total_time += time_frame  # logging

        # if action was vaild, move and update demand:
        if not done:
            vehicle.set_current_coord_to_dest()  # 'jump' to destination
            self.simple_demand_satisfaction(customer_idx=int(actions))  # set demand to zero

        # check if all demands are satisfied,
        # if so return to depot
        if self.check_demands() and not done:
            vehicle.set_node_as_destination(0) # set destination to the single depot with index 0
            time_frame, error_signal = vehicle.calc_move_time(check_if_dest_reachable=True)

            if not np.isnan(time_frame):
                reward -= time_frame
                self.temp_db.total_time += time_frame  # logging

            if not error_signal:
                vehicle.set_current_coord_to_dest()  # 'jump' to destination
                reward += 10 # adding the additional reward, so the reward always stays positive for correct actions
            done = True

        return done, reward


class TSPDroneAction(CustomAction):


    def __init__(self, temp_db, simulation):
        super().__init__()
        self.temp_db = temp_db
        self.simulation = simulation

    def reset(self):
        self.truck_action_waits = False
        self.drone_action_waits = False

    def action_space(self):
        return spaces.MultiDiscrete([self.temp_db.num_customers,self.temp_db.num_customers])

    def _truck_action(self):
        truck = self.temp_db.base_groups['vehicles'][0]  # vehicle with index 0

    def _drone_action(self):
        drone = self.temp_db.base_groups['vehicles'][1]  # vehicle with index 1

    def _truck_transporting_drone_action(self):
        truck = self.temp_db.base_groups['vehicles'][0]  # vehicle with index 0
        drone = self.temp_db.base_groups['vehicles'][1]  # vehicle with index 1

    def decode_actions(self, actions):
        truck = self.temp_db.base_groups['vehicles'][0]  # vehicle with index 0
        drone = self.temp_db.base_groups['vehicles'][1]  # vehicle with index 1

        #action_0 = int(input('Enter truck action: '))
        #action_1 = int(input('Enter drone action: '))

        #actions = np.array([action_0, action_1])

        #print(actions)

        reward = 0
        done_truck = False
        done_drone = False
        drone_is_transported = False

        # Truck:
        if not self.truck_action_waits:
            time_frame_truck, reward_truck, done_truck = self.to_customer(
                customer_idx=int(actions[0]),
                vehicle=truck,
                terminate_on_mistake=False,
                mistake_penalty=0,
                additional_reward=0,
            )
            reward += reward_truck
            self.temp_db.time_till_fin[0] = time_frame_truck
            #print('time_frame_truck',time_frame_truck)
            #print('reward_truck',reward_truck)
            #print('done_truck',done_truck)

        # Drone:
        if not self.drone_action_waits:
            time_frame_drone, reward_drone, done_drone = self.to_customer(
                customer_idx=int(actions[1]),
                vehicle=drone,
                terminate_on_mistake=False,
                mistake_penalty=0,
                additional_reward=0,
            )
            reward += reward_drone
            self.temp_db.time_till_fin[1] = time_frame_drone
            #print('time_frame_drone', time_frame_drone)
            #print('reward_drone', reward_drone)
            #print('done_drone', done_drone)

        # drone and truck are at the same node and have same destination
        if (self.temp_db.status_dict['v_dest'][0] == self.temp_db.status_dict['v_dest'][1]).all() and (
                self.temp_db.status_dict['v_coord'][0] == self.temp_db.status_dict['v_coord'][1]).all():
            drone_is_transported = True
            self.temp_db.time_till_fin[1] = np.copy(self.temp_db.time_till_fin[0])
            #print('drone is transported')

        # Terminate episode if mistakes were made
        if done_truck or done_drone or np.isnan(self.temp_db.time_till_fin[0]) or np.isnan(self.temp_db.time_till_fin[1]):
            return True, reward - 10 # additional terminal penalty

        if not drone_is_transported:
            cur_vehicle_idx = np.argmin(self.temp_db.time_till_fin)
        else:
            cur_vehicle_idx = 0

        time_frame = self.temp_db.time_till_fin[cur_vehicle_idx] # get current timeframe
        self.temp_db.time_till_fin[cur_vehicle_idx] = 0

        if cur_vehicle_idx == 0:
            vehicle = truck
            self.truck_action_waits = False

            if drone_is_transported:
                self.drone_action_waits = False
                self.temp_db.time_till_fin[1] = 0
            else:
                self.drone_action_waits = True
                self.temp_db.time_till_fin[1] = self.temp_db.time_till_fin[1] - time_frame

        else:
            vehicle = drone
            self.drone_action_waits = False
            self.truck_action_waits = True
            self.temp_db.time_till_fin[0] = self.temp_db.time_till_fin[1] - time_frame

        if self.temp_db.status_dict['v_items'][1] != 0 and self.temp_db.status_dict['n_items'][int(actions[cur_vehicle_idx]) + 1]:
            reward += 10

        vehicle.set_current_coord_to_dest()  # 'jump' to destination
        if cur_vehicle_idx == 0:
            self.simple_demand_satisfaction(customer_idx=int(actions[0]))  # set demand to zero

            if drone_is_transported:
                drone.set_current_coord_to_dest()  # 'jump' transported drone to destination

        else:
            if self.temp_db.status_dict['v_items'][1] != 0: # if drone has cargo
                self.simple_demand_satisfaction(customer_idx=int(actions[1]))
                self.temp_db.status_dict['v_items'][1] = 0 # drone has only one cargo slot


        if time_frame == 0:
            cur_vehicle_idx = np.argmax(self.temp_db.time_till_fin)
            time_frame = self.temp_db.time_till_fin[cur_vehicle_idx]  # get current timeframe
            self.temp_db.time_till_fin[cur_vehicle_idx] = 0

            if cur_vehicle_idx == 0:
                vehicle = truck
                self.truck_action_waits = False

                if drone_is_transported:
                    self.drone_action_waits = False
                    self.temp_db.time_till_fin[1] = 0
                else:
                    self.drone_action_waits = False
                    self.temp_db.time_till_fin[1] = self.temp_db.time_till_fin[1] - time_frame

            else:
                vehicle = drone
                self.drone_action_waits = False
                self.truck_action_waits = False
                self.temp_db.time_till_fin[0] = self.temp_db.time_till_fin[1] - time_frame

            if self.temp_db.status_dict['v_items'][1] != 0 and self.temp_db.status_dict['n_items'][int(actions[cur_vehicle_idx]) + 1]:
                reward += 10

            vehicle.set_current_coord_to_dest()  # 'jump' to destination
            if cur_vehicle_idx == 0:
                self.simple_demand_satisfaction(customer_idx=int(actions[0]))  # set demand to zero

                if drone_is_transported:
                    drone.set_current_coord_to_dest()  # 'jump' transported drone to destination

            else:
                if self.temp_db.status_dict['v_items'][1] != 0: # if drone has cargo
                    self.simple_demand_satisfaction(customer_idx=int(actions[1]))
                    self.temp_db.status_dict['v_items'][1] = 0 # drone has only one cargo slot


        reward -= time_frame
        self.temp_db.total_time += time_frame  # logging

        # truck and drone at the same location, drone gets full cargo
        if (self.temp_db.status_dict['v_coord'][0] == self.temp_db.status_dict['v_coord'][1]).all():
            self.temp_db.status_dict['v_items'][1] = 1
            drone.range_restr.reset() # reset range of drone

        # terminate if all demands are satisfied
        if self.check_demands():

            if (self.temp_db.status_dict['v_coord'][0] == self.temp_db.status_dict['v_coord'][1]).all():
                truck.set_node_as_destination(0)  # set destination to the single depot with index 0
                drone.set_node_as_destination(0)  # set destination to the single depot with index 0
                time_frame, error_signal = truck.calc_move_time(check_if_dest_reachable=True) # only truck needs to be checked

                if error_signal or np.isnan(time_frame):
                    return True, reward - 10

                truck.set_current_coord_to_dest() # move
                drone.set_current_coord_to_dest() # move

                return True, reward - time_frame + 10

            else:

                # Try to move drone to depot,
                # assuming this is always better:
                drone.set_node_as_destination(0)
                time_frame_drone, error_signal = drone.calc_move_time(check_if_dest_reachable=True)

                if not error_signal: # drone is able move to depot on its own
                    truck.set_node_as_destination(0)
                    time_frame_truck, error_signal = truck.calc_move_time(
                        check_if_dest_reachable=True)  # truck moves also to depot on its own

                    if error_signal or np.isnan(time_frame_drone) or np.isnan(time_frame_truck):
                        return True, reward - 10

                    time_frame = np.max([time_frame_truck, time_frame_drone])
                    truck.set_current_coord_to_dest()  # move
                    drone.set_current_coord_to_dest()  # move

                    return True, reward - time_frame + 10

                else: # drone has to be transported by truck:
                    self.temp_db.status_dict['v_dest'][0] = np.copy(self.temp_db.status_dict['v_coord'][1])
                    time_frame_truck_to_drone, error_signal = truck.calc_move_time(check_if_dest_reachable=True)

                    if error_signal or np.isnan(time_frame_truck_to_drone):
                        return True, reward - 10

                    truck.set_current_coord_to_dest()

                    truck.set_node_as_destination(0)  # set destination to the single depot with index 0
                    drone.set_node_as_destination(0)  # set destination to the single depot with index 0

                    time_frame, error_signal = truck.calc_move_time(
                        check_if_dest_reachable=True)  # only truck needs to be checked

                    if error_signal or np.isnan(time_frame):
                        return True, reward - 10

                    truck.set_current_coord_to_dest()  # move
                    drone.set_current_coord_to_dest()  # move
                    return True, reward - time_frame - time_frame_truck_to_drone  + 10

        return False, reward


class DiscreteAction:

    def __init__(
            self,
            temp_db,
            key,
            n=None,
    ):

        self.temp_db = temp_db
        self.key = key
        self.n = n

    def finish_init(self):
        if self.n is None:
            self.n = len(self.temp_db(self.key))

    def gym_space(self):
        return spaces.Discrete(self.n)

    def to_node(self, action):

        action = int(action)
        cur_node_coord = self.temp_db.get_val('n_coord')

        if self.temp_db.get_val('n_items')[action] == 0:
            self.temp_db.bestrafung = -10
            #self.temp_db.done = True
            #self.temp_db.bestrafung = -0.01 * self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]]
            #self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]] += 1
            #print(-100)
        if self.temp_db.get_val('n_items')[action] == 1:
            self.temp_db.bestrafung = 10
            #print(100)
        else:
            self.temp_db.bestrafung = -10
            #self.temp_db.bestrafung = -0.01 * self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]]
            #self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]] += 1
            #print(0)

        return cur_node_coord[action]


class BoxAction:

    def __init__(
            self,
            temp_db,
            key,
    ):
        self.temp_db = temp_db
        self.key = key

class BaseActDecoder:

    def __init__(self, act_params, temp_db, simulator):

        self.temp_db = temp_db
        self.simulator = simulator

        [setattr(self, k, v) for k, v in act_params.items()]

        all_outputs = ['coord', 'nodes','move', 'amount', 'v_amount', 'v_to_load', 'load_unload', 'v_load_unload', 'load', 'unload', 'v_load', 'v_unload', 'v_and_single_v', 'v_and_multi_v']

        binary_outputs = ['move','load_unload','v_load_unload','load_sep_unload','v_load_sep_unload']

        value_outputs = ['coord', 'nodes', 'amount','v_amount', 'load_sep_unload', 'v_load_sep_unload', 'v_to_load_index']

        coord_outputs = ['coord', 'nodes']

        if len(list(set(self.contin_outputs) & set(self.discrete_outputs) & set(self.binary_discrete) & set(self.binary_contin))) > 0:
            raise Exception(list(set(self.contin_outputs) & set(self.discrete_outputs) & set(self.binary_discrete) & set(self.binary_contin))+' were dublicates, but must only be used once as outputs.')

        self.val_output_set = set(self.contin_outputs+self.discrete_outputs)
        self.binary_output_set = set(self.binary_contin+self.binary_discrete)

        self.discrete_set = set(self.discrete_outputs+self.binary_discrete)
        self.contin_set = set(self.contin_outputs+self.binary_contin)


        if 'amount' in self.val_output_set:
            if 'load_sep_unload' in self.val_output_set:
                raise Exception('"amount" and "load_sep_unload" can not be both value outputs, set "load_sep_unload" to binary.')

        if 'v_amount' in self.val_output_set:
            if 'v_load_sep_unload' in self.val_output_set:
                raise Exception('"v_amount" and "v_load_sep_unload" can not be both value outputs, set "v_load_sep_unload" to binary.')

        for elem in list(self.val_output_set):
            if elem not in set(value_outputs):
                raise Exception('{} is not accepted as value output, use any of: {}'.format(elem, value_outputs))

        for elem in list(self.binary_output_set):
            if elem not in set(binary_outputs):
                raise Exception('{} is not accepted as value output, use any of: {}'.format(elem, binary_outputs))

        if 'load_sep_unload' in self.binary_output_set  and 'load_unload' in self.binary_output_set:
            raise Exception("'load_sep_unload' and 'load_unload' can't be both binary outputs")

        if 'v_load_sep_unload' in self.binary_output_set  and 'v_load_unload' in self.binary_output_set:
            raise Exception("'v_load_sep_unload' and 'v_load_unload' can't be both binary outputs")

        if 'v_and_single_v' in self.val_output_set  and 'v_and_multi_v' in self.val_output_set:
            raise Exception("'v_and_single_v' and 'v_and_multi_v' can't be both outputs")


        self.act_spaces = []
        self.discrete_bins = np.array([])
        self.discrete_max_val = np.array([])
        self.contin_max_val = np.array([])

        self.discrete_keys = []
        self.contin_keys = []

        
        self.func_dict = {}
        self.check_dict = {
            'coord_bool': True,
            'load_bool': True,
            'unload_bool': True,
            'v_load_bool': True,
            'v_unload_bool': True,
        }

        self.value_dict = {
            'coord': None,
            'load': None,
            'unload': None,
            'v_load': None,
            'v_unload': None,
        }

    def finish_init(self):

        self.init_coord_act(self.val_output_set, self.binary_output_set)
        self.init_cargo_act(self.val_output_set, self.binary_output_set)
        self.init_v_transport_act(self.val_output_set, self.binary_output_set)

        self.index_dict = {}
        all_keys = self.discrete_keys + self.contin_keys
        for i in range(len(all_keys)):
            if all_keys[i] is not None:
                if isinstance(all_keys[i], (list, tuple, np.ndarray)):
                    for elem in all_keys[i]: self.index_dict[elem] = i
                else:
                    self.index_dict[all_keys[i]] = i


    def action_space(self):


        if self.multiple_action_spaces:
            spaces_list = []
            if len(self.contin_max_val) > 0:
                spaces_list.append(spaces.Box(low=0,high=1,shape=(len(self.contin_max_val),)))

            for n in self.discrete_bins:
                spaces_list.append(spaces.Discrete(int(n)))

            return spaces.Tuple(tuple(spaces_list))

        else:
            if len(self.contin_max_val) > 0:
                return spaces.Box(low=0, high=1, shape=(len(self.contin_max_val),))

            for n in self.discrete_bins:
                return spaces.Discrete(int(n))



    def prep_action(self, name, max_val, key=None, act_func=None):

        if name in self.discrete_set:

            if isinstance(max_val, (list, tuple, np.ndarray)):
                for elem in max_val:

                    self.discrete_bins = np.append(self.discrete_bins, min(elem, self.num_discrete_bins))
                    self.discrete_max_val = np.append(self.discrete_max_val, elem)
            else:
                self.discrete_bins = np.append(self.discrete_bins, min(max_val, self.num_discrete_bins))
                self.discrete_max_val = np.append(self.discrete_max_val, max_val)
            self.discrete_keys.append(key)

        else:
            if isinstance(max_val, (list, tuple, np.ndarray)):
                for elem in max_val:
                    self.contin_max_val = np.append(self.contin_max_val, elem)
            else:
                self.contin_max_val = np.append(self.contin_max_val, max_val)
            self.contin_keys.append(key)
        
        if act_func is not None:
            if isinstance(key, (list, tuple, np.ndarray)):
                for k in key:
                    self.func_dict[k] = act_func
            else:
                self.func_dict[key] = act_func

        #self.index_dict[key] = self.index_dict[name]


    def init_coord_act(self, val_output_set, binary_output_set):
        '''
        coordinates:
        - no coordinates -> automate movement
        - only coordinates
        - only nodes
        - both coordinates and nodes -> reward based on nearest node (option: move to node or move to coordinates?)
        - additionaly move
        '''

        # Binary addition:
        if 'move' in binary_output_set:
            self.prep_action('move', 2, 'coord_bool', self.binary_check)


        # both coordinates and nodes -> reward based on nearest node (option: move to node or move to coordinates?)
        if 'coord' in val_output_set and 'nodes' in val_output_set:
            self.prep_action('coord', [self.temp_db.grid[0], self.temp_db.grid[1]], 'compare_coord', None)
            self.prep_action('nodes', self.temp_db.num_nodes, 'coord', self.to_node)
            self.func_dict['compare_coord'] = self.compare_coord


        # only coordinates:
        elif 'coord' in val_output_set:
            self.prep_action('coord', [self.temp_db.grid[0], self.temp_db.grid[1]], 'coord', self.two_values)

        # only nodes:
        elif 'nodes' in val_output_set:
            self.prep_action('nodes', self.temp_db.num_nodes, 'coord', self.to_node)

        # automate:
        else:
            self.func_dict['coord'] = self.auto_value

    
    def init_cargo_act(self, val_output_set, binary_output_set):
        '''
        cargo:
        - no outputs -> automate cargo
        - only amount -> automate loading, unloading based on current location
        - only load_sep_unload as value outputs -> no automation

        - additions if 'amount' as value output:
            - 'load_sep_unload' as TWO binary outputs -> no automation
            - alternative 'load_unload' as ONE binary output -> automate loading/unloading

        - additions if 'load_sep_unload' as value output:
            - alternative 'load_unload' as ONE binary output -> no automation
        '''

        # binary additions:
        if 'load_sep_unload' in binary_output_set:
            self.prep_action('load_sep_unload', 2, 'load_bool', self.binary_check)
            self.prep_action('load_sep_unload', 2, 'unload_bool', self.binary_check)

        elif 'load_unload' in binary_output_set:
            self.prep_action('load_sep_unload', 2, ['load_bool','unload_bool'], self.binary_check)

        # only 'amount'
        if 'amount' in val_output_set:
            max_val = max(self.temp_db.min_max_dict['load'][1], self.temp_db.outputs_max['unload'])
            self.prep_action('amount', max_val, ['load','unload'], self.one_value)


        # only 'load_sep_unload'
        elif 'load_sep_unload' in val_output_set:
            self.prep_action('load_sep_unload', self.temp_db.min_max_dict['load'][1], 'load', self.one_value)
            self.prep_action('load_sep_unload', self.temp_db.min_max_dict['unload'][1], 'unload', self.one_value)


        # automate
        else:
            self.func_dict['load'] = self.auto_value
            self.func_dict['unload'] = self.auto_value
        

            
    def init_v_transport_act(self, val_output_set, binary_output_set):
        '''
        same as cargo but additionaly vehicle to load can be chosen:
        - 'v_and_single_v' chooses single vehicle (one output)
        - 'v_and_multi_v' chooses multiple vehicles (multi output contin, same outputs for discrete but not one hotted)
        '''

        
        '''
        elif 'v_and_multi_v' in val_output_set:
            v_load_cargo_funcs.append(self.v_multi_v)
            v_unload_cargo_funcs.append(self.v_multi_v)
        '''


                # binary additions:
        if 'v_load_sep_unload' in binary_output_set:
            self.prep_action('v_load_sep_unload', 2, 'v_load_bool', self.binary_check)
            self.prep_action('v_load_sep_unload', 2, 'v_unload_bool', self.binary_check)

        elif 'v_load_unload' in binary_output_set:
            self.prep_action('v_load_sep_unload', 2, ['v_load_bool','v_unload_bool'], self.binary_check)


        if 'v_amount' in val_output_set:
            self.prep_action('v_amount', self.temp_db.outputs_max['v_unload'], 'v_unload', self.one_value)
        else:
            self.func_dict['v_unload'] = self.auto_value
        '''
        # only 'amount'
        if 'v_amount' in val_output_set:
            self.prep_action('v_amount', !!!!!!!, ['v_load','v_unload'], self.one_value)

        
        # only 'load_sep_unload'
        elif 'v_load_sep_unload' in val_output_set:
            self.prep_action('v_load_sep_unload', !!!!!!!, 'v_load', self.one_value)
            self.prep_action('v_load_sep_unload', !!!!!!!, 'v_unload', self.one_value)
        '''

        # automate
        

        # specifying the vehicle to load
        if 'v_to_load_index' in val_output_set:
            self.prep_action('v_to_load_index', self.temp_db.outputs_max['v_load'], 'v_load', self.one_value)
        else:
            self.func_dict['v_load'] = self.auto_value



    def binary_check(self, key):
        self.check_dict[key] = bool(self.actions[self.index_dict[key]])


    def one_value(self, key):
        if self.check_dict[key+'_bool'] == True:
            self.value_dict[key] = self.actions[self.index_dict[key]]


    def two_values(self, key):
        if self.check_dict[key+'_bool'] == True:
            self.value_dict[key] = np.array([self.actions[self.index_dict[key]-1], self.actions[self.index_dict[key]]])


    def compare_coord(self, key):
        chosen_coord = np.array([self.actions[self.index_dict[key]-1], self.actions[self.index_dict[key]]])
        real_coord = self.value_dict['coord']
        self.temp_db.action_signal['compare_coord'][self.temp_db.v_index] -= np.sum(np.abs(real_coord-chosen_coord))


    def to_node(self, key):

        if self.check_dict[key+'_bool'] == True:

            cur_node_coord = self.temp_db.get_val('n_coord')

            if self.temp_db.get_val('n_items')[int(self.actions[self.index_dict[key]])] == 0:
                self.temp_db.bestrafung = -10
                #self.temp_db.done = True
                #self.temp_db.bestrafung = -0.01 * self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]]
                #self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]] += 1
                #print(-100)
            if self.temp_db.get_val('n_items')[int(self.actions[self.index_dict[key]])] == 1:
                self.temp_db.bestrafung = 10
                #print(100)
            else:
                self.temp_db.bestrafung = -10
                #self.temp_db.bestrafung = -0.01 * self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]]
                #self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]] += 1
                #print(0)
            self.value_dict[key] = cur_node_coord[int(self.actions[self.index_dict[key]])]

    def auto_value(self, key):
        if self.check_dict[key+'_bool'] == True:
            self.value_dict[key] = None

    def decode_discrete(self, actions):
        return actions[0]
        #for i in range(len(actions)):
            #actions[i] = (actions[i] / (self.discrete_bins[i]-1)) - 1

        #return np.round(actions*self.discrete_max_val).astype(int)

    def decode_contin(self, actions):
        return np.round(actions*(self.contin_max_val-1)).astype(int)


    def decode_actions(self, actions):
        if self.temp_db.status_dict['v_free'][self.temp_db.cur_v_index] == 1:

            if not isinstance(actions, np.ndarray):
                actions = np.array([actions])

            if len(self.discrete_max_val) != 0: self.actions = self.decode_discrete(actions[:len(self.discrete_max_val)]).ravel()
            if len(self.contin_max_val) != 0: self.actions = self.decode_contin(actions[-len(self.contin_max_val):]).ravel()

            [self.func_dict[key](key) for key in self.func_dict.keys()]

            if self.check_dict['v_unload_bool']: self.simulator.unload_vehicle(self.value_dict['v_unload'])
            if self.check_dict['v_load_bool']:   self.simulator.load_vehicle(self.value_dict['v_load'])
            if self.check_dict['load_bool']:     self.simulator.load_items(self.value_dict['load'])

            if self.check_dict['coord_bool']:    self.simulator.set_destination(self.value_dict['coord'])
            if self.check_dict['unload_bool']:   self.simulator.unload_items(self.value_dict['unload'])


            
            #self.simulator.recharge_range(self.temp_db.v_index)
            
            #self.temp_db.signals_dict['v_free'][self.temp_db.cur_v_index] += 1

        #else:
            #self.temp_db.signals_dict['v_free'][self.temp_db.cur_v_index] -= 1


