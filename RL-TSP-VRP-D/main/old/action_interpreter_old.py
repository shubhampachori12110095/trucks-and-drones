import numpy as np

def discrete_to_norm(dimension_list, action_list):
    '''Returns the normalized actions of a discrete action space.'''
    return [(action_list[i]/(dimension_list[i]-1)) for i in range(len(action_list))]



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
