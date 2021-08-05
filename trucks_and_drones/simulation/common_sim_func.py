'''

'''
import numpy as np
# Used for obj creation:
# ----------------------------------------------------------------------------------------------------------------

def param_interpret(var):
    if isinstance(var, (list, tuple, np.ndarray)):
        if len(var) == 2:
            return np.random.randint(var[0],var[1]+1)
    return var

def max_param_val(var):
    if isinstance(var, (list, tuple, np.ndarray)):
        return np.max(var)
    return var

def random_coordinates(grid):
    return (np.random.randint(0,grid[0]+1), np.random.randint(0,grid[1]+1))

def return_indices_of_a(list_a, list_b):
    return [i for i, v in enumerate(list_a) if v in set(list_b)]

def l_ignore_none(l):
	return [i for i in l if i is not None]

def clip_pos(value):
	if value < 0:
		value = 0
	return value

#def compare_coordinates
