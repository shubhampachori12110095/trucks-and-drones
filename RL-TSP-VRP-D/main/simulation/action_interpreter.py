import numpy as np
from gym import spaces


def output_parameter(
        mode             = 'single_vehicle', # 'multi_vehicle'
        flattened        = 'per_output', #'per_vehicle', #'all'
        contin_outputs   = ['coord','amount','v_amount'],
        discrete_outputs = ['nodes', 'v_to_load'],
        binary_discrete  = ['move', 'load_unload', 'v_load_unload'],
        binary_contin    = [],
        discrete_bins    = 20,
        combine          = 'contin', # 'discrete', 'by_categ', 'all', list of lists of output names
        ):
    return {
        'contin_outputs': contin_outputs,
        'discrete_outputs': discrete_outputs,
        'discrete_dims': discrete_dims,
        'combine': combine,
    }



class BaseActionInterpreter:

    def __init__(self, v_action_dict, action_prio_list, simulator, only_at_node_interactions=False):

        self.temp_db = simulator.temp_db
        self.simulator = simulator

        all_outputs = ['coord', 'nodes','move', 'amount', 'v_amount', 'v_to_load', 'load_unload', 'v_load_unload', 'load', 'unload', 'v_load', 'v_unload', 'v_and_single_v', 'v_and_multi_v']

        binary_outputs = ['move','load_unload','v_load_unload','load_sep_unload','v_load_sep_unload']

        value_outputs = ['amount','v_amount', 'load_sep_unload', 'v_load_sep_unload', 'v_to_load_index']

        coord_outputs = ['coord', 'nodes']

        if len(list(set(self.contin_outputs) & set(self.discrete_outputs) & set(self.binary_discrete) & set(self.binary_contin))) > 0:
            raise Exception(list(set(self.contin_outputs) & set(self.discrete_outputs) & set(self.binary_discrete) & set(self.binary_contin))+' were dublicates, but must only be used once as outputs.')

        val_output_set = set(self.contin_outputs+self.discrete_outputs)
        binary_output_set = set(self.binary_contin+self.binary_discrete)

        self.discrete_set = set(self.discrete_outputs+self.binary_discrete)
        self.contin_set = set(self.contin_outputs+self.binary_contin)
        

        if 'amount' in val_output_set:
            if 'load_sep_unload' in val_output_set:
                raise Exception('"amount" and "load_sep_unload" can not be both value outputs, set "load_sep_unload" to binary.')

        if 'v_amount' in val_output_set:
            if 'v_load_sep_unload' in val_output_set:
                raise Exception('"v_amount" and "v_load_sep_unload" can not be both value outputs, set "v_load_sep_unload" to binary.')

        for elem in list(val_output_set):
            if elem not in set(value_outputs):
                raise Exception(elem+' is not accepted as value output, use any of: '+value_outputs)

        for elem in list(inary_output_set):
            if elem not in set(binary_outputs):
                raise Exception(elem+' is not accepted as binary output, use any of: '+binary_outputs)

        if 'load_sep_unload' in binary_output_set  and 'load_unload' in binary_output_set:
            raise Exception("'load_sep_unload' and 'load_unload' can't be both binary outputs")

        if 'v_load_sep_unload' in binary_output_set  and 'v_load_unload' in binary_output_set:
            raise Exception("'v_load_sep_unload' and 'v_load_unload' can't be both binary outputs")

        if 'v_and_single_v' in value_outputs  and 'v_and_multi_v' in value_outputs:
            raise Exception("'v_and_single_v' and 'v_and_multi_v' can't be both outputs")


        self.act_spaces = []
        self.discrete_bins = np.array([])
        self.discrete_max_val = np.array([])
        self.contin_max_val = np.array([])

        
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

        self.init_coord_act(val_output_set, binary_output_set)
        self.init_cargo_act(val_output_set, binary_output_set)
        self.init_v_transport_act(val_output_set, binary_output_set)

        self.index_dict = {}
        all_keys = self.discrete_keys+self.contin_keys
            for i in range(len(all_keys)):
                if all_keys[i] is not None:
                    if isinstance(all_keys[i], (list, tuple, np.ndarray)):
                        for elem in all_keys[i]: self.index_dict[elem] = i
                    else:
                        self.index_dict[all_keys[i]] = i



    def prep_action(name, max_val, key=None, act_func=None):
        
        if name in self.discrete_set:
            
            if isinstance(max_val, (list, tuple, np.ndarray)):
                for elem in max_val:
                    self.discrete_bins = np.append(self.discrete_bins, min(elem, self.discrete_bins))
                    self.discrete_max_val = np.append(self.discrete_max_val, elem)
            else:
                self.discrete_bins = np.append(self.discrete_bins, min(max_val, self.discrete_bins))
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
            self.func_dict[key] = act_func


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
        if 'coord' in val_output_set and 'node' in val_output_set:
            self.prep_action('coord', [self.temp_db.grid[0], self.temp_db.grid[1]], 'compare_coord', None)
            self.prep_action('nodes', self.temp_db.num_nodes, 'coord', self.to_node)
            self.func_dict['compare_coord'] = self.compare_coord


        # only coordinates:
        elif 'coord' in val_output_set:
            self.prep_action('coord', [self.temp_db.grid[0], self.temp_db.grid[1]], 'coord', self.two_values)

        # only nodes:
        elif 'node' in val_output_set:
            self.prep_action('node', self.temp_db.num_nodes, 'coord', self.to_node)

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
            self.prep_action('amount', !!!!!!!, ['load','unload'], self.one_value)


        # only 'load_sep_unload'
        elif 'load_sep_unload' in val_output_set:
            self.prep_action('load_sep_unload', !!!!!!!, 'load', self.one_value)
            self.prep_action('load_sep_unload', !!!!!!!, 'unload', self.one_value)


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
            self.prep_action('v_amount', !!!!!!!, 'v_unload', self.one_value)
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
        else:
            self.func_dict['v_unload'] = self.auto_value

        # specifying the vehicle to load
        if 'v_to_load_index' in val_output_set:
            self.prep_action('v_to_load_index', !!!!!!!, 'v_load', self.one_value)
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
            cur_node_coord = self.temp_db.status_dict['d_coord']+self.temp_db.status_dict['c_coord']
            self.value_dict[key] = cur_node_coord[self.actions[self.index_dict[key]]]


    def auto_value(self, key):
        if self.check_dict[key+'_bool'] == True:
            self.value_dict[key] = None


    def decode_discrete(self, actions):
        for i in range(len(actions)):
            actions[i] = np.argmax(actions[i]) / (self.discrete_bins[i]-1)

        return np.round(actions*self.discrete_max_val).astype(int)


    def decode_contin(self, actions):
        return np.round(actions*self.contin_max_val).astype(int)


    def take_actions(self, actions):
        if self.temp_db.status_dict['v_free'][vehicle_i] == 1:

            if len(self.discrete_max_val) != 0: self.actions = decode_discrete(actions[:len(self.discrete_max_val)]).ravel()
            if len(self.contin_max_val) != 0: self.actions = decode_contin(actions[-len(self.contin_max_val):]).ravel()

            [self.func_dict[key](key) for key in self.func_dict.keys()]

            if self.value_dict['coord_bool']:    self.simulator.move(self.temp_db.v_index, self.value_dict['coord'])
            if self.value_dict['unload_bool']:   self.simulator.unload_cargo(self.temp_db.v_index, None, self.value_dict['unload'])
            if self.value_dict['load_bool']:     self.simulator.load_cargo(self.temp_db.v_index, None, self.value_dict['load'])
            if self.value_dict['v_unload_bool']: self.simulator.unload_vehicles(self.temp_db.v_index, self.value_dict['v_unload'])
            if self.value_dict['v_load_bool']:   self.simulator.load_vehicles(self.temp_db.v_index, self.value_dict['v_load'])
            
            self.simulator.recharge_range(self.temp_db.v_index)
            self.temp_db.action_signal['v_free'][i] += 1

        else:
            self.temp_db.action_signal['v_free'][i] -= 1


