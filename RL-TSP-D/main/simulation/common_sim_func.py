'''

'''
import numpy as np
# Used for obj creation:
# ----------------------------------------------------------------------------------------------------------------

def param_interpret(variable):
    if isinstance(variable, (list, tuple, np.ndarray)):
        if len(variable) == 2:
            return np.random.randint(variable[0],variable[1]+1)
    return variable

def random_coordinates(grid):
    return (np.random.randint(0,grid[0]+1), np.random.randint(0,grid[1]+1))

def return_indices_of_a(list_a, list_b):
    return [i for i, v in enumerate(list_a) if v in set(list_b)]

def compare_coordinates
