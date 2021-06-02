import numpy as np
from gym import spaces

def None_to_empty_list(variable):
    if isinstance(variable, (list, tuple, np.ndarray)):
        return variable
    elif variable is None:
        return []
    elif isinstance(variable, str):
        return [variable]
    else:
        raise Exception(variable+" wasn't a list, tuple, ndarray, string or None")


def input_parameter(
        image_input       = ['grid'],
        contin_inputs     = ['coordinates','values','vehicles','customers','depots'],
        discrete_inputs   = ['binary'],
        discrete_dims     = 20,
        combine_per_index = ['per_vehicle', 'per_customer', 'per_depot'], # list of input name lists
        combine_per_type  = None,
        # Flattens per combined (and all inputs not in a combined list),
        # if no combination are used everything will be flattened,
        flatten           = True,
        flatten_images    = False,
        ):
    return {
        'image_input'      : image_input,
        'contin_inputs'    : contin_inputs,
        'discrete_inputs'  : discrete_inputs,
        'discrete_dims'    : discrete_dims,
        'combine_per_index': combine_per_index,
        'combine_per_type' : combine_per_type,
        'flatten'          : flatten,
        'flatten_images'   : flatten_images,
        }


class BaseStateInterpreter:

    def __init__(self, input_param, visualizor, simulator):

        # Init parameter:
        [setattr(self, k, None_to_empty_list(v)) for k, v in input_param.items()]

        # Init objects:
        self.visualizor = visualizor
        self.temp_db    = simulator.temp_db

        
        # Init dicts:
        self.dict_of_inputs = {
            'contin_dict'   :{},
            'discrete_dict' :{}
            }

        # Prepare Inputs to use:
        self.all_inputs = self.discrete_inputs + self.contin_inputs

        self.discrete_coord  = list(set(self.discrete_inputs) & set(self.temp_db.key_groups_dict['coordinates']))
        self.discrete_binary = list(set(self.discrete_inputs) & set(self.temp_db.key_groups_dict['binary']))
        self.discrete_value  = list(set(self.discrete_inputs) & set(self.temp_db.key_groups_dict['values']))

        self.contin_coord  = list(set(self.discrete_inputs) & set(self.temp_db.key_groups_dict['coordinates']))
        self.contin_binary = list(set(self.discrete_inputs) & set(self.temp_db.key_groups_dict['binary']))
        self.contin_value  = list(set(self.discrete_inputs) & set(self.temp_db.key_groups_dict['values']))


        # Prepare input combinations to use
        for elem in self.combine_per_index:
            if elem == 'per_vehicle':
                elem = list(set(self.all_inputs) & set(self.temp_db.key_groups_dict['vehicles']))
            elif elem == 'per_customer':
                elem = list(set(self.all_inputs) & set(self.temp_db.key_groups_dict['customers']))
            elif elem == 'per_depot':
                elem = list(set(self.all_inputs) & set(self.temp_db.key_groups_dict['depots']))
        

        all_combined_input_keys = sum(self.combine_per_index + self.combine_per_type)
        self.uncombined_elem    = np.setdiff1d(self.all_inputs, all_combined_input_keys)

        self.combine_per_type += self.uncombined_elem


    def coord_to_contin(self, key):
        ''' Normalizes list of Coordinates'''
        coord_list = self.temp_db.status_dict[key]

        array_x = np.array([elem[0]/self.temp_db.grid[0] for elem in coord_list])
        array_y = np.array([elem[1]/self.temp_db.grid[1] for elem in coord_list])

        return np.concatenate((array_x, array_y), axis=1)


    def value_to_contin(self, key):
        ''' Normalizes list of Values'''
        max_value = self.temp_db.max_values_dict[key]
        min_value = self.temp_db.min_values_dict[key]
        value_list = self.temp_db.status_dict[key]
        return (np.array(value_list) - min_value) // (max_value - min_value)


    def coord_to_discrete(self, key):
        ''' Converts list of Coordinates to discrete'''
        coord_list = self.temp_db.status_dict[key]
        
        array_x = np.zeros((len(coord_list), self.temp_db.grid[0]))
        array_y = np.zeros((len(coord_list), self.temp_db.grid[1]))

        array_x[np.arange(len(coord_list)), np.array([int(elem[0]) for elem in coord_list])] = 1
        array_y[np.arange(len(coord_list)), np.array([int(elem[1]) for elem in coord_list])] = 1

        return np.concatenate((array_x, array_y), axis=1)


    def binary_to_discrete(self, key):
        ''' Converts list of binary values to discrete'''
        binary_list = self.temp_db.status_dict[key]
        
        array_binary = np.zeros((len(binary_list), 2))
        array_binary[np.arange(len(binary_list)), np.array(binary_list)] = 1

        return array_binary


    def value_to_discrete(self, key):
        ''' Converts list of Values to discrete'''
        value_list = self.temp_db.status_dict[key]
        
        array_value = np.zeros((len(value_list), self.discrete_dims))

        max_value = self.temp_db.max_values_dict[key]
        min_value = self.temp_db.min_values_dict[key]

        values = (np.array(value_list) - min_value) // (max_value - min_value) * (self.discrete_dims - 1)

        array_value[np.arange(len(value_list)), values] = 1

        return array_value


    def combine_index(self, keys_list):
        ''' Creates a list of inputs for each index (e.g. vehicles, customers or depots).
        Note that each element of inputs_list needs to be the same lenght.'''

        inputs_list =  [self.dict_of_dicts['contin_dict'][key]   for key in keys_list]
        inputs_list += [self.dict_of_dicts['discrete_dict'][key] for key in keys_list]
        
        list_of_arrays = [np.array([])]**len(inputs_list[0])
        
        for input_array in inputs_list:
            for i in range(len(input_array)):
                list_of_arrays[i] = np.append(list_of_arrays[i], input_array[i])

        return list_of_arrays


    def combine_type(self, keys_list):

        inputs_list =  [self.dict_of_dicts['contin_dict'][key]   for key in keys_list]
        inputs_list += [self.dict_of_dicts['discrete_dict'][key] for key in keys_list]
        
        list_of_arrays = [np.array([])]**len(inputs_list[0])

        return inputs_list


    def combined_flatten(self, to_combine):
        ''' Combines and flattens list, tuples of arrays or a single array to one dimensional array'''
        if isinstance(to_combine, list) and len(to_combine) > 1:
            return np.concatenate((to_combine), axis=None)
        elif isinstance(to_combine, list):
            to_combine = to_combine[0]
        return np.ravel(to_combine)


    def observe_state(self):

        for key in self.contin_coord : self.dict_of_dicts['contin_dict'][key] = self.coord_to_contin(key)
        for key in self.contin_binary: self.dict_of_dicts['contin_dict'][key] = np.array(self.temp_db.status_dict[key])
        for key in self.contin_value : self.dict_of_dicts['contin_dict'][key] = self.value_to_contin(key)

        for key in self.discrete_coord : self.dict_of_dicts['discrete_dict'][key] = self.coord_to_discrete(key)
        for key in self.discrete_binary: self.dict_of_dicts['discrete_dict'][key] = self.binary_to_discrete(key)
        for key in self.discrete_value : self.dict_of_dicts['discrete_dict'][key] = self.value_to_discrete(key)


        inputs_by_indeces = [self.combine_index(keys_list) for keys_list in self.combine_per_index]
        inputs_by_types   = [self.combine_type(keys_list) for keys_list in self.combine_per_type]

        all_inputs = inputs_by_indeces + inputs_by_types

        if self.flatten:
            all_inputs = [self.combined_flatten(elem) for elem in all_inputs]

        for key in self.image_input:
            image_array = self.visualizor.convert_to_img_array().transpose([1, 0, 2]) // 255
            if self.flatten_images:
                image_array = self.combined_flatten(image_array)
            all_inputs  = [image_array] + all_inputs

        if len(all_inputs) == 1:
                return all_inputs[0]
        return all_inputs


    def obs_space(self):

        all_inputs = self.observe_state()
        if isinstance(all_inputs, np.ndarray):
            return spaces.Box(low=0, high=1, shape=np.shape(all_inputs), dtype=np.uint8)
        if isinstance(all_inputs, list):
            return [spaces.Box(low=0, high=1, shape=np.shape(elem), dtype=np.uint8) for elem in all_inputs]



