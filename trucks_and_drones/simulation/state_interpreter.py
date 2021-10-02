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
        return variable

def flatten_list(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


class BaseObsEncoder:

    def __init__(self, obs_params, temp_db, visualizor):

        # Init parameter:
        [setattr(self, k, None_to_empty_list(v)) for k, v in obs_params.items()]

        # Init objects:
        self.visualizor = visualizor
        self.temp_db    = temp_db

        
        # Init dicts:
        self.contin_dict = {}
        self.discrete_dict = {}

        for i in range(len(self.contin_inputs)):
            if self.contin_inputs[i] == 'coordinates':
                self.contin_inputs[i] = self.temp_db.key_groups_dict['coordinates']
            elif self.contin_inputs[i] == 'vehicles':
                self.contin_inputs[i] = self.temp_db.key_groups_dict['vehicles']
            elif self.contin_inputs[i] == 'customers':
                self.contin_inputs[i] = self.temp_db.key_groups_dict['customers']
            elif self.contin_inputs[i] == 'depots':
                self.contin_inputs[i] = self.temp_db.key_groups_dict['depots']
            elif self.contin_inputs[i] == 'binary':
                self.contin_inputs[i] = self.temp_db.key_groups_dict['binary']
            elif self.contin_inputs[i] == 'restrictions':
                self.contin_inputs[i] = self.temp_db.key_groups_dict['restrictions']
            elif self.contin_inputs[i] == 'values':
                self.contin_inputs[i] = self.temp_db.key_groups_dict['values']

        for i in range(len(self.discrete_inputs)):
            if self.discrete_inputs[i] == 'coordinates':
                self.discrete_inputs[i] = self.temp_db.key_groups_dict['coordinates']
            elif self.discrete_inputs[i] == 'vehicles':
                self.discrete_inputs[i] = self.temp_db.key_groups_dict['vehicles']
            elif self.discrete_inputs[i] == 'customers':
                self.discrete_inputs[i] = self.temp_db.key_groups_dict['customers']
            elif self.discrete_inputs[i] == 'depots':
                self.discrete_inputs[i] = self.temp_db.key_groups_dict['depots']
            elif self.discrete_inputs[i] == 'binary':
                self.discrete_inputs[i] = self.temp_db.key_groups_dict['binary']
            elif self.discrete_inputs[i] == 'restrictions':
                self.discrete_inputs[i] = self.temp_db.key_groups_dict['restrictions']
            elif self.discrete_inputs[i] == 'values':
                self.discrete_inputs[i] = self.temp_db.key_groups_dict['values']

        # Prepare Inputs to use:
        self.all_inputs = flatten_list(self.discrete_inputs) + flatten_list(self.contin_inputs)

        self.discrete_coord  = list(set(flatten_list(self.discrete_inputs)) & set(self.temp_db.key_groups_dict['coordinates']))
        self.discrete_binary = list(set(flatten_list(self.discrete_inputs)) & set(self.temp_db.key_groups_dict['binary']))
        self.discrete_value  = list(set(flatten_list(self.discrete_inputs)) & set(self.temp_db.key_groups_dict['values']))

        self.contin_coord  = list(set(flatten_list(self.contin_inputs)) & set(self.temp_db.key_groups_dict['coordinates']))
        self.contin_binary = list(set(flatten_list(self.contin_inputs)) & set(self.temp_db.key_groups_dict['binary']))
        self.contin_value  = list(set(flatten_list(self.contin_inputs)) & set(self.temp_db.key_groups_dict['values']))

        # Prepare input combinations to use
        for i in range(len(self.combine_per_index)):
            if self.combine_per_index[i] == 'per_vehicle':
                self.combine_per_index[i] = list(set(self.all_inputs) & set(self.temp_db.key_groups_dict['vehicles']))
            elif self.combine_per_index[i] == 'per_customer':
                self.combine_per_index[i] = list(set(self.all_inputs) & set(self.temp_db.key_groups_dict['customers']))
            elif self.combine_per_index[i] == 'per_depot':
                self.combine_per_index[i] = list(set(self.all_inputs) & set(self.temp_db.key_groups_dict['depots']))


        all_combined_input_keys = flatten_list(self.combine_per_index) + flatten_list(self.combine_per_type)
        self.uncombined_elem    = list(np.setdiff1d(self.all_inputs, all_combined_input_keys))

        self.combine_per_type = self.combine_per_type + self.uncombined_elem

    def reset(self):
        pass


    def coord_to_contin(self, key):
        ''' Normalizes list of Coordinates'''
        coord_list = self.temp_db.get_val(key)

        array_x = np.array([elem[0]/self.temp_db.grid[0] for elem in coord_list])
        array_y = np.array([elem[1]/self.temp_db.grid[1] for elem in coord_list])

        if key == 'c_coord':
            for i in range(self.temp_db.num_customers):
                if self.temp_db.status_dict['n_items'][i + self.temp_db.num_depots] == 0:
                    array_x[i] = 0.0
                    array_y[i] = 0.0

        return np.nan_to_num(np.append(array_x, array_y))


    def value_to_contin(self, key):
        ''' Normalizes list of Values'''

        max_value = self.temp_db.min_max_dict[key][1]
        min_value = self.temp_db.min_max_dict[key][0]
        value_list = self.temp_db.get_val(key)
        
        if max_value is None:
            return np.ones((len(value_list)))
        if (max_value - min_value) == 0:
            return np.zeros((len(value_list)))
        return np.nan_to_num((np.array(value_list) - min_value) / (max_value - min_value))


    def coord_to_discrete(self, key):
        ''' Converts list of Coordinates to discrete'''
        coord_list = self.temp_db.get_val(key)

        array_x = np.zeros((len(coord_list), self.temp_db.grid[0]+1))
        array_y = np.zeros((len(coord_list), self.temp_db.grid[1]+1))

        array_x[np.arange(len(coord_list)), np.array([int(elem[0]) for elem in coord_list])] = 1
        array_y[np.arange(len(coord_list)), np.array([int(elem[1]) for elem in coord_list])] = 1

        return np.nan_to_num(np.concatenate((array_x, array_y), axis=1))


    def binary_to_discrete(self, key):
        ''' Converts list of binary values to discrete'''
        binary_list = self.temp_db.get_val(key)

        array_binary = np.zeros((len(binary_list), 2))
        array_binary[np.arange(len(binary_list)), np.array(binary_list, dtype=np.int)] = 1

        return np.nan_to_num(array_binary)


    def value_to_discrete(self, key):
        ''' Converts list of Values to discrete'''
        value_list = self.temp_db.get_val(key)

        print(value_list)
        print(self.temp_db.min_max_dict[key][1])
        print(self.temp_db.min_max_dict[key][0])

        array_value = np.zeros((len(value_list), self.discrete_dims))

        max_value = self.temp_db.min_max_dict[key][1]
        min_value = self.temp_db.min_max_dict[key][0]
        
        if max_value is None:
            values = np.ones((len(value_list))) * (self.discrete_dims - 1)
        else:
            values = (np.array(value_list) - min_value) // (max_value - min_value) * (self.discrete_dims - 1)

        array_value[np.arange(len(value_list)), values] = 1

        return np.nan_to_num(array_value)


    def combine_index(self, keys_list):
        ''' Creates a list of inputs for each index (e.g. vehicles, customers or depots).
        Note that each element of inputs_list needs to be the same lenght.'''

        inputs_list =  [self.contin_dict[key]   for key in keys_list if key in set(self.contin_dict.keys())]
        inputs_list += [self.discrete_dict[key] for key in keys_list if key in set(self.discrete_dict.keys())]

        if len(inputs_list) > 0:

            list_of_arrays = [np.array([]) for i in range(len(inputs_list[0]))]
            
            for input_array in inputs_list:
                print(input_array)
                for i in range(len(input_array)):
                    print(list_of_arrays, input_array)
                    list_of_arrays[i] = np.nan_to_num(np.append(list_of_arrays[i], input_array[i]))

            return list_of_arrays


    def combine_type(self, keys_list):

        inputs_list =  [self.contin_dict[key]   for key in keys_list if key in set(self.contin_dict.keys())]
        inputs_list += [self.discrete_dict[key] for key in keys_list if key in set(self.discrete_dict.keys())]
        
        
        if len(inputs_list) > 0:
            #list_of_arrays = [np.array([]) for i in range(len(inputs_list[0]))]
            return inputs_list


    def combined_flatten(self, to_combine):
        ''' Combines and flattens list, tuples of arrays or a single array to one dimensional array'''
        if isinstance(to_combine, list) and len(to_combine) > 1:
            return np.concatenate((to_combine), axis=None)
        elif isinstance(to_combine, list):
            to_combine = to_combine[0]
        return np.nan_to_num(np.ravel(to_combine))



    def observe_state(self):

        for key in self.contin_coord : self.contin_dict[key] = self.coord_to_contin(key)
        for key in self.contin_binary: self.contin_dict[key] = np.array(self.temp_db.get_val(key))
        for key in self.contin_value : self.contin_dict[key] = self.value_to_contin(key)

        for key in self.discrete_coord : self.discrete_dict[key] = self.coord_to_discrete(key)
        for key in self.discrete_binary: self.discrete_dict[key] = self.binary_to_discrete(key)
        for key in self.discrete_value : self.discrete_dict[key] = self.value_to_discrete(key)

        #inputs_by_indeces = [self.combine_index(keys_list) for keys_list in self.combine_per_index]
        #inputs_by_types   = [self.combine_type(keys_list) for keys_list in self.combine_per_type]

        #all_inputs = inputs_by_indeces + inputs_by_types

        all_inputs = [self.contin_dict[key] for key in self.contin_dict.keys()]
        all_inputs += [self.discrete_dict[key] for key in self.discrete_dict.keys()]

        if self.flatten:
            all_inputs = [self.combined_flatten(elem) for elem in all_inputs]

        for key in self.image_input:
            image_array = 1 - (self.visualizor.convert_to_img_array().transpose([1, 0, 2]) / 255)
            if self.flatten_images:
                image_array = self.combined_flatten(image_array)
            all_inputs  = [image_array] + all_inputs

        if self.output_as_array:
            return np.concatenate(all_inputs)

        if len(all_inputs) == 1:
                return all_inputs[0]
        return all_inputs


    def obs_space(self):

        all_inputs = self.observe_state()

        if isinstance(all_inputs, np.ndarray):
            return spaces.Box(
                -np.ones((len(all_inputs))).astype(np.float32),
                np.ones((len(all_inputs))).astype(np.float32),
                dtype=np.float32)

        if isinstance(all_inputs, list):
            return spaces.Tuple(tuple([spaces.Box(low=0.0, high=1.0, shape=np.shape(elem)) for elem in all_inputs]))



