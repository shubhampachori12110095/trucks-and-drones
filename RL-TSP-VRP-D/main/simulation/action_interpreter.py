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





def discrete_to_norm(dimension_list, action_list):
    '''Returns the normalized actions of a discrete action space.'''
    return [(action_list[i]/(dimension_list[i]-1)) for i in range(len(action_list))]


class BaseActionInterpreter:

    def __init__(self, v_action_dict, action_prio_list, simulator, only_at_node_interactions=False):

        self.temp_db = simulator.temp_db
        self.simulator = simulator

        all_outputs = ['coord', 'nodes','move', 'amount', 'v_amount', 'v_to_load', 'load_unload', 'v_load_unload', 'load', 'unload', 'v_load', 'v_unload', 'v_and_single_v', 'v_and_multi_v']

        binary_outputs = ['move','load_unload','v_load_unload','load_sep_unload','v_load_sep_unload']

        value_outputs = ['amount','v_amount', 'load_sep_unload', 'v_load_sep_unload', 'v_and_single_v', 'v_and_multi_v']

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


        self.shapes = []
        self.bins = []

    def decode_actions(self, actions, max_val_array):
        # check if actions are discrete
        for i in range(len(actions)):
            if self.bins[i] != 0:
                actions[i] = np.argmax(actions[i]) / (self.bins[i]-1)
            else:
                actions[i] = actions[i][0]

        return np.round(actions*max_val_array).astype(int)

    def calc_num_bins(self, key, max_val):
        
        if key in self.discrete_set:
            self.bins.append(min(max_val,self.discrete_bins))
            self.shapes.append(min(max_val,self.discrete_bins))
        else:
            self.bins.append(0)
            self.shapes.append(1)


    def init_coord_indeces(self, val_output_set, binary_output_set):
        '''
        coordinates:
        - no coordinates -> automate movement
        - only coordinates
        - only nodes
        - both coordinates and nodes -> reward based on nearest node (option: move to node or move to coordinates?)
        - additionaly move
        '''

        self.coord_funcs = []

        # Binary addition:
        if 'move' in binary_output_set:
            self.coord_funcs.append(self.binary_check)
            self.bins.append(self.calc_num_bins('move',2))

        # both coordinates and nodes -> reward based on nearest node (option: move to node or move to coordinates?)
        if 'coord' in val_output_set and 'node' in val_output_set:
            self.coord_funcs.append(self.compare_coord)
            self.coord_funcs.append(self.to_node)
            self.bins.append(self.calc_bins_and_shapes('coord',self.temp_db.grid[0]))
            self.bins.append(self.calc_num_bins('coord',self.temp_db.grid[1]))
            self.bins.append(self.calc_num_bins('nodes',self.temp_db.num_nodes))


        # only coordinates:
        elif 'coord' in val_output_set:
            self.coord_funcs.append(self.to_coordinates)
            self.bins.append(self.calc_num_bins('coord',self.temp_db.grid[0]))
            self.bins.append(self.calc_num_bins('coord',self.temp_db.grid[1]))

        # only nodes:
        elif 'node' in val_output_set:
            self.coord_funcs.append(self.to_node)
            self.bins.append(self.calc_num_bins('nodes',self.temp_db.num_nodes))

        # automate:
        else:
            self.coord_funcs.append(self.auto_coordinates)

    
    def init_cargo_indices(self, val_output_set, binary_output_set):
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

        self.load_cargo_funcs = []
        self.unload_cargo_funcs = []

        # binary additions:
        if 'load_sep_unload' in binary_output_set:
            self.load_cargo_funcs.append(self.two_binary_check)
            self.unload_cargo_funcs.append(self.two_binary_check)

        elif 'load_unload' in binary_output_set:
            self.load_cargo_funcs.append(self.binary_check)
            self.unload_cargo_funcs.append(self.binary_check)


        # only 'amount'
        if 'amount' in val_output_set:
            self.load_cargo_funcs.append(self.one_amount)
            self.unload_cargo_funcs.append(self.one_amount)

        # only 'load_sep_unload'
        elif 'load_sep_unload' in val_output_set:
            self.load_cargo_funcs.append(self.two_amounts)
            self.unload_cargo_funcs.append(self.two_amounts)

        # automate
        else:
            self.load_cargo_funcs.append(self.auto_amounts)
            self.unload_cargo_funcs.append(self.auto_amounts)

            
    def init_v_transport_indices(self, val_output_set, binary_output_set):
        '''
        same as cargo but additionaly vehicle to load can be chosen:
        - 'v_and_single_v' chooses single vehicle (one output)
        - 'v_and_multi_v' chooses multiple vehicles (multi output contin, same outputs for discrete but not one hotted)
        '''

        v_load_cargo_funcs = []
        v_unload_cargo_funcs = []

        # specifying the vehicle to load/unload
        if 'v_and_single_v' in val_output_set:
            v_load_cargo_funcs.append(self.v_single_v)
            v_unload_cargo_funcs.append(self.v_single_v)
        
        elif 'v_and_multi_v' in val_output_set:
            v_load_cargo_funcs.append(self.v_multi_v)
            v_unload_cargo_funcs.append(self.v_multi_v)


        # binary additions:
        if 'load_sep_unload' in binary_output_set:
            v_load_cargo_funcs.append(self.two_binary_check)
            v_unload_cargo_funcs.append(self.two_binary_check)

        elif 'load_unload' in binary_output_set:
            v_load_cargo_funcs.append(self.binary_check)
            v_unload_cargo_funcs.append(self.binary_check)


        # only 'v_amount'
        if 'v_amount' in val_output_set:
            v_load_cargo_funcs.append(self.one_amount)
            v_unload_cargo_funcs.append(self.one_amount)

        # only 'v_load_sep_unload'
        elif 'v_load_sep_unload' in val_output_set:
            v_load_cargo_funcs.append(self.two_amounts)
            v_unload_cargo_funcs.append(self.two_amounts)

        # automate
        else:
            v_load_cargo_funcs.append(self.auto_amounts)
            v_unload_cargo_funcs.append(self.auto_amounts)

####################################################################################################################################################




        coord_binary = list({'move'} & binary_output_set)
        coord_value = list({'coord', 'nodes'} & val_output_set)
        if len(coord_value) == 0:
            coord_value.append('auto_coord')
        elif len(coord_value) == 2:
            coord_value.append('compare_coord')

        cargo_binary = list({'load_sep_unload','load_unload'} & binary_output_set)
        cargo_value = list({'load_sep_unload','amount'} & val_output_set)
        if len(cargo_value) == 0:
            coord_value.append('auto_amount')

        v_to_load = list({'v_and_single_v','v_and_multi_v'} & val_output_set)
        v_cargo_binary = list({'v_load_sep_unload','v_load_unload'} & binary_output_set)
        v_cargo_value = list({'v_load_sep_unload','v_amount'} & val_output_set)
        if len(v_cargo_value) == 0:
            coord_value.append('v_auto_amount')

        self.func_dict = {}
        for elem in list({'move','load_unload','v_load_unload'} & binary_output_set)
            self.func_dict[elem] = self.binary_check
        for elem in list({'load_sep_unload','v_load_sep_unload'} & binary_output_set)
            self.func_dict[elem] = self.multi_binary_check
        for elem in list({'load_sep_unload','v_load_sep_unload'} & val_output_set)
            self.func_dict[elem] = self.single_amount


    def binary_check(self)
        self.finished = bool(self.cur_list[self.cur_index])

    def multi_binary_check(self):
        if self.cur_list[self.cur_index][0] + self.cur_list[self.cur_index][1] == 0:
            self.finished = True
        elif self.cur_list[self.cur_index][0] + self.cur_list[self.cur_index][1] == 2:
            self.load_unload_mode = None
        else:
            self.load_unload_mode = np.argmax(self.cur_list[self.cur_index])

    def multi_amount(self)



    def decode_cargo(self):

        self.load_unload_mode = None


    def decode_coordinates(self, cur_list):

        for elem in 

        if move:

            if coord and node:
                compare_coord

        self.cur_index = 0
        self.cur_list  = cur_list


        for key in coord_binary:
            self.binary_check
        


        self.simulator.move(vehicle_i, coordinates)























        self.init_coord_functions(val_output_set, binary_output_set)
        self.init_cargo_functions(val_output_set, binary_output_set)
        self.init_v_transport_functions(val_output_set, binary_output_set)


    def init_coord_index(self, val_output_set, binary_output_set):
        '''
        coordinates:
        - no coordinates -> automate movement
        - only coordinates
        - only nodes
        - both coordinates and nodes -> reward based on nearest node (option: move to node or move to coordinates?)
        - additionaly move
        '''

        self.coord_funcs = []

        # Binary addition:
        if 'move' in binary_output_set:
            self.coord_funcs.append(self.binary_check)

        # both coordinates and nodes -> reward based on nearest node (option: move to node or move to coordinates?)
        if 'coord' in val_output_set and 'node' in val_output_set:
            self.coord_funcs.append(self.compare_coord)

        # only coordinates:
        elif 'coord' in val_output_set:
            self.coord_funcs.append(self.to_coordinates)

        # only nodes:
        elif 'node' in val_output_set:
            self.coord_funcs.append(self.to_node)

        # automate:
        else:
            self.coord_funcs.append(self.auto_coordinates)

    
    def init_cargo_functions(self, val_output_set, binary_output_set):
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

        self.load_cargo_funcs = []
        self.unload_cargo_funcs = []

        # binary additions:
        if 'load_sep_unload' in binary_output_set:
            self.load_cargo_funcs.append(self.two_binary_check)
            self.unload_cargo_funcs.append(self.two_binary_check)

        elif 'load_unload' in binary_output_set:
            self.load_cargo_funcs.append(self.binary_check)
            self.unload_cargo_funcs.append(self.binary_check)


        # only 'amount'
        if 'amount' in val_output_set:
            self.load_cargo_funcs.append(self.one_amount)
            self.unload_cargo_funcs.append(self.one_amount)

        # only 'load_sep_unload'
        elif 'load_sep_unload' in val_output_set:
            self.load_cargo_funcs.append(self.two_amounts)
            self.unload_cargo_funcs.append(self.two_amounts)

        # automate
        else:
            self.load_cargo_funcs.append(self.auto_amounts)
            self.unload_cargo_funcs.append(self.auto_amounts)

            
    def init_v_transport_functions(self, val_output_set, binary_output_set):
        '''
        same as cargo but additionaly vehicle to load can be chosen:
        - 'v_and_single_v' chooses single vehicle (one output)
        - 'v_and_multi_v' chooses multiple vehicles (multi output contin, same outputs for discrete but not one hotted)
        '''

        v_load_cargo_funcs = []
        v_unload_cargo_funcs = []

        # specifying the vehicle to load/unload
        if 'v_and_single_v' in val_output_set:
            v_load_cargo_funcs.append(self.v_single_v)
            v_unload_cargo_funcs.append(self.v_single_v)
        
        elif 'v_and_multi_v' in val_output_set:
            v_load_cargo_funcs.append(self.v_multi_v)
            v_unload_cargo_funcs.append(self.v_multi_v)


        # binary additions:
        if 'load_sep_unload' in binary_output_set:
            v_load_cargo_funcs.append(self.two_binary_check)
            v_unload_cargo_funcs.append(self.two_binary_check)

        elif 'load_unload' in binary_output_set:
            v_load_cargo_funcs.append(self.binary_check)
            v_unload_cargo_funcs.append(self.binary_check)


        # only 'v_amount'
        if 'v_amount' in val_output_set:
            v_load_cargo_funcs.append(self.one_amount)
            v_unload_cargo_funcs.append(self.one_amount)

        # only 'v_load_sep_unload'
        elif 'v_load_sep_unload' in val_output_set:
            v_load_cargo_funcs.append(self.two_amounts)
            v_unload_cargo_funcs.append(self.two_amounts)

        # automate
        else:
            v_load_cargo_funcs.append(self.auto_amounts)
            v_unload_cargo_funcs.append(self.auto_amounts)


    def binary_check(self, key):
        if binary_dict[key] == 0:
            val_dict[key] == None


    def two_binary_check(self, key):
        if binary_dict[key][0] == 0:
            val_dict[key][0] == None
        
        if binary_dict[key][1] == 0:
            val_dict[key][1] == None


    def one_amount(self, key):


    def two_amounts(self, key):

    
    def auto_amount(self, key):






































# Actions per Vehicle Parameter:
# ----------------------------------------------------------------------------------------------------------------
'''
All parameter can be defined as single int for CONTIN and DISCRETE data,

for DISCRETE the outputs will be expanded by definded dimension in the discrete_dict:
num_outputs*num_dimension

if you are using a network thats only capable to take one discrete action,
the total number of outputs will be num_outputs*num_dimension*(sum used outputs)
(in this case its recommended to only use num_nodes_outputs and num_v_to_load_outputs with one loadable vehicle per transpoter)

for both CONTIN and DISCRETE defining an output as 0 means not using it.
'''
def create_v_num_outputs(
        ############# NONE ergänzen ###############
        # coordinates:
        num_coord_outputs = None ERGÄNZEN,
        
        # alternative to num_coord, to choose a discrete node
        # when used both the agent will be forced to learn the coord of a corresponding node
        num_nodes_outputs = 0,

        # can be used to extend the discrete nodes,
        # if only_at_node_interactions=False, vehicles can also meet at NON nodes coordinates
        # will also force the agent to learn the coord of vehicles, if used with num_coord
        num_v_to_load_outputs = 0,
        
        # determines if the vehicle should move or not:
        num_move_outputs = 0,
        
        # base amount of cargo or vehicles to load or unload
        num_amount_outputs = 0,

        # can used without num_amount_outputs, if you only want an action for vehicle to vehicle interactions
        # determines the number of vehicles to unload or load from the vehicle
        # if num_amount is NOT 0, then base num_amount will be used for cargo
        num_v_amount_outputs = 0,

        # extension or alternative to num_amount (and num_v_amount if this is 0),
        # seperates loading and unloading, if num_amount is 0 this will determine the amount
        num_load_outputs = 0,
        num_unload_outputs = 0,

        # extension or alternative to num_v_amount,
        # seperates vehicle loading and unloading, if num_v_amount is 0 this will determine the amount
        num_v_load_outputs = 0,
        num_v_unload_outputs = 0,

        seperate_actions = True ERGÄNZEN,
    ):
    return {
        'num_coord_outputs':     num_coord_outputs,
        'num_nodes_outputs':     num_nodes_outputs,
        'num_v_to_load_outputs': num_v_to_load_outputs,
        'num_move_outputs':      num_move_outputs,
        'num_amount_outputs':    num_amount_outputs,
        'num_v_amount_outputs':  num_v_amount_outputs,
        'num_load_outputs':      num_load_outputs,
        'num_unload_outputs':    num_unload_outputs,
        'num_v_load_outputs':    num_v_load_outputs,
        'num_v_unload_outputs':  num_v_unload_outputs,
        }


def examples_define_action_parameter():
    ## Examples for simple discrete and contin actions with no transportable vehicles:

    # Example DISCRETE actions with 10 nodes:
    # the total outputs will be num_nodes_outputs*10, so we have to set num_nodes_outputs=1
    discrete_nodes_no_transp_v = create_v_num_outputs(num_nodes_outputs=[1,10])

    # Example DISCRETE actions with coordinates of a grid=[10,15]
    # one output with a dimension of 10 and one with 15:
    discrete_coord__no_transp_v = create_v_num_outputs(num_coord_outputs=[2,15])

    # Example for CONTIN with 10 nodes and a grid=[10,15]
    # the single output will be transformed with int(output*10) to get the node index
    contin_nodes_no_transp_v = create_v_num_outputs(num_nodes_outputs=1)
    # one output with a dimension of 10 and one with 15:
    contin_coord__no_transp_v = create_v_num_outputs(num_coord_outputs=2)




# Action Interpretation:
# ----------------------------------------------------------------------------------------------------------------

def discrete_to_norm(dimension_list, action_list):
    '''Returns the normalized actions of a discrete action space.'''
    return [(action_list[i]/(dimension_list[i]-1)) for i in range(len(action_list))]


class MoveFunctions:
    '''The ActionInterpreter will use this functions based on the set parameters to calculate coordinates.'''

    def __init__(placeholder_dict, simulator, auto_move_mode='random'):

        self.placeholder_dict = placeholder_dict
        self.simulator        = simulator

    def compare_coord(self):
        ''' Compares the chosen coordinates with the coordinates of a chosen node.'''
        for i in range(len(self.placeholder_dict['coord_outputs'])):
            ###### cur_coord_nodes ist manchmal NONE!!!!!!!!!!!!!!!!!!!!!!!!!!
            real_coord   = self.simulator.temp_db.cur_coord_nodes[self.placeholder_dict['nodes_outputs'][i]]
            chosen_coord = self.placeholder_dict['coord_outputs'][i]
            self.simulator.temp_db.action_signal['compare_coord'][i] += (real_coord - ((real_coord-chosen_coord) * 2))

        self.coord_list = self.placeholder_dict['coord_outputs']

    def move_and_coord(self):
        ''' Checks if the vehicle should move to the chosen coordinates.'''
        for i in range(len(self.placeholder_dict['move_outputs'])):
            if self.placeholder_dict['move_outputs'][i] == 0:
                self.placeholder_dict['coord_outputs'][i] = None

        self.coord_list = self.placeholder_dict['coord_outputs']

    def move_and_nodes(self):
        ''' Checks if the vehicle should move to the chosen node'''

        for i in range(len(self.placeholder_dict['move_outputs'])):
            if self.placeholder_dict['move_outputs'][i] == 0:
                self.placeholder_dict['nodes_outputs'][i] = None

        ###### cur_coord_nodes ist manchmal NONE!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.coord_list = [self.simulator.temp_db.cur_coord_nodes[self.placeholder_dict['nodes_outputs'][i]] for i in self.placeholder_dict['nodes_outputs']]

    def random_nodes(self):
        ''' sets the coord_list to shuffled node coordinates.'''
        ######################### ERGÄNZE RESTRIKTION DASS ER NICHT ZURÜCKFÄHRT!!!!!
        ###### cur_coord_nodes ist manchmal NONE!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.coord_list = random.sample(elem.cur_coord for elem in self.simulator.temp_db.cur_coord_nodes)

    def nearest_nodes(self):
        ''' sets the coord_list to nearest nodes'''
        ######################### ERGÄNZE RESTRIKTION DASS ER NICHT ZURÜCKFÄHRT!!!!! 

        # init coordinates list
        self.coordinates = []
        # copy coordinates of all nodes to list
        ###### cur_coord_nodes ist manchmal NONE!!!!!!!!!!!!!!!!!!!!!!!!!!
        node_coordinates = [elem.cur_coord for elem in self.simulator.temp_db.cur_coord_nodes]
        # iterate through each vehicle
        for i in range(len(self.simulator.temp_db.base_groups['vehicles'])):
            # Check if vehicle is free to move:
            if any('vehicle_'+str(i) in elem for elem in self.simulator.temp_db.free_vehicles):
                # calculate distance to all nodes
                distance_list = [elem - self.simulator.temp_db.base_groups['vehicles'][i].cur_coord for elem in node_coordinates]
                # transform distance by summation of x and y to absulte values
                distance_list = [abs(sum(elem)) for elem in distance_list]
                # add coordinates to list by indexing the min distance
                self.coordinates.append(node_coordinates[argmin(distance_list)])
            else:
                # append None to signal no movement:
                self.coordinates.append(None)

    def auto_movement(self):

        if self.auto_move_mode == 'random':
            # create coordinates to random nodes
            self.random_nodes()
        elif self.auto_move_mode == 'nearest_neighbour'
            # create coordintaes to nearest nodes
            self.nearest_nodes()

    def base_move_func(self):
        ''' Calls simulator.move to move all vehicles that should be moved'''
        # create list of indices for coordinates that aren't None
        coord_index_list = [i for i, elem in enumerate(self.coord_list) if elem != None]
        # Move vehicles based on coordinates that aren't None
        self.simulator.move(i,self.coord_list[i]) for i in coord_index_list

    



def create_move_functions(key_list, placeholder_dict, simulator):
    
    obj = MoveFunctions(placeholder_dict, simulator)
    function_list = []
    
    if set(['coord_outputs','nodes_outputs']).issubset(set(key_list)):
        function_list.append(obj.compare_coord)

    if set(['move_outputs']).issubset(set(key_list)):

        if set(['coord_outputs']).issubset(set(key_list)):
            function_list.append(obj.move_and_coord)

        elif set(['nodes_outputs']).issubset(set(key_list)):
            function_list.append(obj.move_and_nodes)
        
        else:
            function_list.append(obj.auto_movement)

    elif not set(['coord_outputs']).issubset(set(key_list)) or set(['move_outputs']).issubset(set(key_list)):
        function_list.append(obj.auto_movement)

    function_list.append(obj.base_move_func)
    return function_list



class CargoAndVehiclesFunctions:

    def __init__(placeholder_dict, simulator):

        self.placeholder_dict = placeholder_dict
        self.simulator        = simulator

        self.num_vehicles     = #####################len(self.simulator.temp_db.base_groups['vehicles'])


    def only_v_amount(self):
        self.num_v_list = self.placeholder_dict['v_amount_outputs']

    def only_amount(self):
        self.cargo_amount = self.placeholder_dict['amount_outputs']


    def v_to_load(self):
        self.chosen_v_list = set(self.num_v_list).intersection(self.placeholder_dict['v_to_load_outputs'])


    def v_amount_and_v_unload(self):
        self.num_v_list = [a*b for a,b in zip(self.placeholder_dict['v_amount_outputs'],self.placeholder_dict['v_unload_outputs'])]

    def v_amount_and_v_load(self):
        self.num_v_list = [a*b for a,b in zip(self.placeholder_dict['v_amount_outputs'],self.placeholder_dict['v_load_outputs'])]

    def amount_and_unload(self):
        self.cargo_amount = [a*b for a,b in zip(self.placeholder_dict['amount_outputs'],self.placeholder_dict['unload_outputs'])]

    def amount_and_load(self):
        self.cargo_amount = [a*b for a,b in zip(self.placeholder_dict['amount_outputs'],self.placeholder_dict['load_outputs'])]


    def only_v_unload(self):
        self.num_v_list = self.placeholder_dict['v_unload_outputs']

    def only_v_load(self):
        self.num_v_list = self.placeholder_dict['v_load_outputs']

    def only_unload(self):
        self.cargo_amount = self.placeholder_dict['unload_outputs']

    def only_load(self):
        self.cargo_amount = self.placeholder_dict['load_outputs']


    def auto_amount(self):
        # try to unload/load as much cargo as possible
        self.num_v_list = [self.simulator.temp_db.base_groups['vehicles'][i].cargo_obj.cargo_per_step for i in range(self.num_vehicles)]

    def auto_v_amount(self):
        # only unload/load one vehicle
        self.cargo_amount = [1]*len(self.simulator.temp_db.base_groups['vehicles'])


    def base_unload_vehicles_func(self):
        self.simulator.unload_vehicles(i_and_num[0], i_and_num[1]) for i_and_num in self.v_unloading_num_v_list

    def base_load_vehicles_func(self):
        self.simulator.load_vehicle(i_and_j[0], i_and_j[1]) for i_and_j in v_loading_v_list

    def base_unload_cargo_func(self):
        self.simulator.load_cargo(i_j_a[0], i_j_a[1], i_j_a[2]) for i_j_a in v_unloading_c_amount_list

    def base_load_cargo_func(self):
        self.simulator.load_cargo(i_j_a[0], i_j_a[1], i_j_a[2]) for i_j_a in v_loading_d_amount_list



def create_unload_vehicles_functions(key_list, placeholder_dict, simulator):
    #['amount_outputs','load_outputs'  ,'v_amount_outputs','v_to_load_outputs','v_load_outputs']
    
    obj = CargoAndVehiclesFunctions(placeholder_dict, simulator)
    function_list = []

    if set(['v_to_load_outputs']).issubset(set(key_list)):
        obj
    
    if set(['v_amount_outputs','v_load_outputs']).issubset(set(key_list)):
        function_list.append(obj.v_amount_and_v_unload)

    elif set(['v_amount_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_v_amount)

    elif set(['v_unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_v_unload)
    else:
        no_v_amount_spezified = True

    # cargo amount:
    if set(['amount_outputs','unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.amount_and_unload)

    elif set(['amount_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_amount)

    elif set(['unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_unload)
    else:
        no_amount_spezified = True

    if no_v_amount_spezified and no_amount_spezified:
        return []

    elif no_v_amount_spezified:
        function_list.append(obj.auto_v_amount)

    elif no_amount_spezified:
        function_list.append(obj.auto_amount)

    function_list.append(obj.base_unload_vehicles_func)
    return function_list



def create_load_vehicles_functions(key_list, placeholder_dict, simulator):
    #['amount_outputs','unload_outputs','v_amount_outputs','v_unload_outputs','v_to_load_outputs']
    
    obj = CargoAndVehiclesFunctions(placeholder_dict, simulator)
    function_list = []
    
    if set(['v_amount_outputs','v_unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.v_amount_and_v_unload)

    elif set(['v_amount_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_v_amount)

    elif set(['v_unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_v_unload)
    else:
        no_v_amount_spezified = True

    # cargo amount:
    if set(['amount_outputs','unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.amount_and_unload)

    elif set(['amount_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_amount)

    elif set(['load_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_unload)
    else:
        no_amount_spezified = True

    if no_v_amount_spezified and no_amount_spezified:
        return []

    elif no_v_amount_spezified:
        function_list.append(obj.auto_v_amount)

    elif no_amount_spezified:
        function_list.append(obj.auto_amount)

    if set(['v_to_load_outputs']).issubset(set(key_list)):
        function_list.append(obj.v_to_load)

    function_list.append(obj.base_load_vehicles_func)
    return function_list



def create_unload_cargo_functions(key_list, placeholder_dict, simulator):

    obj = CargoAndVehiclesFunctions(placeholder_dict, simulator)
    function_list = []

    # cargo amount:
    if set(['amount_outputs','unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.amount_and_unload)

    elif set(['amount_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_amount)

    elif set(['unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_unload)
    else:
        function_list.append(obj.auto_amount)

    function_list.append(obj.base_unload_cargo_func)
    return function_list



def create_load_cargo_functions(key_list, placeholder_dict, simulator):

    obj = CargoAndVehiclesFunctions(placeholder_dict, simulator)
    function_list = []

    # cargo amount:
    if set(['amount_outputs','unload_outputs']).issubset(set(key_list)):
        function_list.append(obj.amount_and_unload)

    elif set(['amount_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_amount)

    elif set(['load_outputs']).issubset(set(key_list)):
        function_list.append(obj.only_unload)
    else:
        function_list.append(obj.auto_amount)

    function_list.append(obj.base_load_cargo_func)
    return function_list



class BaseActionInterpreter:

    def __init__(self, v_action_dict, action_prio_list, simulator, only_at_node_interactions=False):

        self.max_value_dict = max_value_dict

        # Delete outputs that arent used:
        self.placeholder_dict = {k:v[0] for k,v in v_action_dict.items() if v[0] != 0}
        self.dimension_dict   = {k:v[1] for k,v in v_action_dict.items() if v[0] != 0}

        # Calculate the total num of outputs per vehicle:
        self.num_outputs_vehicle = sum(self.placeholder_dict.values())

        
        ## Create Action Prio Dict
        ## functions will be executed based on sequence of action_prio_list

        # initilize lists that contain all possible outputs for an action:
        all_move_list            = ['coord_outputs' ,'nodes_outputs' ,'move_outputs'],
        all_unload_vehicles_list = ['amount_outputs','unload_outputs','v_amount_outputs','v_unload_outputs']
        all_load_vehicle_list    = ['amount_outputs','load_outputs'  ,'v_amount_outputs','v_to_load_outputs','v_load_outputs']
        all_unload_cargo_list    = ['amount_outputs','unload_outputs']
        all_load_cargo_list      = ['amount_outputs','load_outputs']
        
        # create empty action_prio_dict:
        self.action_prio_dict = {key:[] for key in action_prio_list}

        # assign the used outputs to action_prio_dict:
        all_relevant_outputs  = self.placeholder_dict.keys()
        self.action_prio_dict['move']            = set(all_move_list).intersection(all_relevant_outputs)
        self.action_prio_dict['unload_vehicles'] = set(all_unload_vehicles_list).intersection(all_relevant_outputs)
        self.action_prio_dict['load_vehicle']    = set(all_load_vehicle_list).intersection(all_relevant_outputs)
        self.action_prio_dict['unload_cargo']    = set(all_unload_cargo_list).intersection(all_relevant_outputs)
        self.action_prio_dict['load_cargo']      = set(all_load_cargo_list).intersection(all_relevant_outputs)


        ## Create functions and assign to dict:
        self.action_func_dict = {
            'move':            create_move_functions(           self.action_prio_dict['move']           , self.placeholder_dict, simulator),
            'unload_vehicles': create_unload_vehicles_functions(self.action_prio_dict['unload_vehicles'], self.placeholder_dict, simulator),
            'load_vehicle':    create_load_vehicle_functions(   self.action_prio_dict['load_vehicle']   , self.placeholder_dict, simulator),
            'unload_cargo':    create_unload_cargo_functions(   self.action_prio_dict['unload_cargo']   , self.placeholder_dict, simulator),
            'load_cargo':      create_load_cargo_functions(     self.action_prio_dict['load_cargo']     , self.placeholder_dict, simulator),
            }

    def action_space(self):

        if self.seperate_actions == True:
            
            action_space_list = []
            for key in self.placeholder_dict.keys():
                
                if self.dimension_dict[key] != None:
                    action_space_list.append(spaces.Discrete(self.dimension_dict[key]))
                else:
                    action_space_list.append(spaces.Box(low=0, high=1, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8))

 


    def outputs_to_actions(self, ouputs_array):

        actions_list = np.array_split(np.squeeze(ouputs_array), self.num_outputs_vehicle)

        self.placeholder_dict.update((key, []) for key in self.placeholder_dict)
        
        for actions in actions_list:
            
            i = 0
            for key in self.placeholder_dict:

                some_actions = actions[i:i+self.num_outputs_dict[key]]
                
                if self.dimension_dict[key] != None: 
                    some_actions = discrete_to_norm(self.dimension_dict[key], actions)

                some_actions = [int(some_actions[i]*self.max_value_dict[key][i]) for i in range(len(some_actions))]

                self.placeholder_dict[key].append(some_actions)

            i+=1

    
    def take_actions(self):
                
        for action_key in self.action_prio_dict:

            key_list          = self.action_prio_dict[action_key]
            some_actions_list = [self.placeholder_dict[k] for k in key_list]
            
            exec_func(some_actions_list) for exec_func in self.action_func_dict[action_key]


    def action_space(self):
        spaces.Discrete(N_DISCRETE_ACTIONS)
