import gym
import numpy as np
from gym import spaces

class CustomAction:

    def __init__(self):
        pass

    def build(self, temp_db, simulation):
        self.temp_db = temp_db
        self.simulation = simulation

    def gym_space(self):
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

        # No demand, mistake:
        if self.temp_db.get_val('n_items')[node_idx] == 0:
            reward += mistake_penalty
            if terminate_on_mistake:
                done = True

        # Customer has demand:
        else:
            reward += additional_reward

        vehicle.set_node_to_destination(node_idx)

        # calculate move time and check if vehicle can actually move:
        time, done_signal = vehicle.calc_move_time(check_if_dest_reachable=True)

        if done_signal:
            reward += mistake_penalty

        if terminate_on_mistake and done_signal:
            done = True

        return time, reward, done



class TSPAction:

    def __init__(self, temp_db, simulation):
        self.temp_db = temp_db
        self.simulation = simulation

    def build(self, temp_db, simulation):
        self.temp_db = temp_db
        self.simulation = simulation

    def action_space(self):
        return spaces.Discrete(self.temp_db.num_nodes - 1)

    def decode_actions(self, actions):

        action = int(actions)  + 1
        cur_node_coord = self.temp_db.get_val('n_coord')

        if self.temp_db.get_val('n_items')[action] == 0:
            self.temp_db.bestrafung = -10
            done = True
            # self.temp_db.done = True
            # self.temp_db.bestrafung = -0.01 * self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]]
            # self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]] += 1
            # print(-100)
        if self.temp_db.get_val('n_items')[action] == 1:
            self.temp_db.bestrafung = 10
            done = False
            # print(100)
        else:
            self.temp_db.bestrafung = -10
            # self.temp_db.bestrafung = -0.01 * self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]]
            # self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]] += 1
            # print(0)

        self._move(action)
        self.temp_db.status_dict['n_items'][action] = 0
        if not done:
            done = bool(np.sum(self.temp_db.get_val('n_items')[1:]) == 0)

        if done:
            self._move(0)

        self.temp_db.total_time += self.temp_db.cur_time_frame

        return done

    def _move(self, node_index):
        coordinates = np.array(self.temp_db.get_val('n_coord')[node_index])
        vehicle = self.temp_db.base_groups['vehicles'][0]
        self.temp_db.status_dict['v_dest'][self.temp_db.cur_v_index] = coordinates
        vehicle.v_move(calc_time=True)
        self.temp_db.time_till_fin[0] = np.nan_to_num(self.temp_db.time_till_fin[0])
        self.temp_db.total_time += self.temp_db.time_till_fin[0]
        self.temp_db.status_dict['v_coord'][vehicle.v_index] = coordinates
        self.temp_db.past_coord_not_transportable_v[vehicle.v_index].append(
            np.copy(self.temp_db.status_dict['v_coord'][vehicle.v_index]))
        self.temp_db.time_till_fin[0] = 0

class TSPDroneAction:

    def __init__(self, temp_db, simulation):
        self.temp_db = temp_db
        self.simulation = simulation

    def build(self, temp_db, simulation):
        self.temp_db = temp_db
        self.simulation = simulation

    def action_space(self):
        return spaces.Tuple(spaces=(
                spaces.Discrete(self.temp_db.num_customers + 1),
                spaces.Discrete(self.temp_db.num_customers + 1)
            )
        )

    def decode_actions(self, actions):

        truck_act = int(actions[0]) + 1
        drone_act = int(actions[1]) + 1
        cur_node_coord = self.temp_db.get_val('n_coord')

        if self.temp_db.get_val('n_items')[action] == 0:
            self.temp_db.bestrafung = -10
            done = True
            # self.temp_db.done = True
            # self.temp_db.bestrafung = -0.01 * self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]]
            # self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]] += 1
            # print(-100)
        if self.temp_db.get_val('n_items')[action] == 1:
            self.temp_db.bestrafung = 10
            done = False
            # print(100)
        else:
            self.temp_db.bestrafung = -10
            # self.temp_db.bestrafung = -0.01 * self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]]
            # self.temp_db.bestrafung_multiplier[self.actions[self.index_dict[key]]] += 1
            # print(0)

        self._move(action)
        self.temp_db.status_dict['n_items'][action] = 0
        if not done:
            done = bool(np.sum(self.temp_db.get_val('n_items')[1:]) == 0)

        if done:
            self._move(0)

        self.temp_db.total_time += self.temp_db.cur_time_frame

        return done


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


