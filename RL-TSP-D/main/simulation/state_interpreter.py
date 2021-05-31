import numpy as np
from gym import spaces


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
def input_parameter(
        # coordinates:
        grid_representation = 'layers',# 'img', 'sum-x-y-layers'
        auxiliary_inputs    = [['cargo_amount', 1], ['v_amount', 1]],
        seperate_inputs     = True,
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


class BaseStateInterpreter:

    def __init__(self, input_param, visualizor, simulator, img_inputs=False):

        [setattr(self, k, v) for k, v in input_param.items()]

        self.visualizor = visualizor
        self.simulator  = simulator


    def obs_space(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)


        