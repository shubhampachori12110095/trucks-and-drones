import numpy as np
import gym
import pandas as pd
import random

import keras
from keras.layers import Input, Dense, Flatten, Concatenate


def net_parameter(
        num_neurons_multip = 2,
        input_shape_list   = [(5,), (6,), (3,)],
        start_indiv_hidden = [
                              [10, 12, 6],
                              [20, 20, 20]
                             ],
        combined_hidden    = [60,
                              80,
                              40],
        end_indiv_hidden   = [
                              [20, 20, 20, 20],
                              [10, 12, 4, 2]
                             ],
        outputs_list       = [5, 6, 2, 1],
        hidden_activation  = 'relu',
        output_activation  = ['softmax', 'softmax', 'softmax', None],
        ):
    return {
        'input_shape_list':   input_shape_list,
        'start_indiv_hidden': start_indiv_hidden,
        'combined_hidden':    combined_hidden,
        'end_indiv_hidden':   end_indiv_hidden,
        'outputs_list':       outputs_list,
        'hidden_activation':  hidden_activation,
        'output_activation':  output_activation,
        }

def transform_input_dict(net_parameter):

    for key in input_param.keys():


def check_env_space(env_space):

    if isinstance(env_space, (list, tuple, np.ndarray)):
        return env_space, len(env_space)
    else:
        return [env_space], 1


class DiamondNetwork:

    def __init__(self, env, net_parameter):
        
        [setattr(self, k, v) for k, v in net_parameter.items()]

        observation_space, len_obs_list = check_env_space(self.env.observation_space)
        action_space, len_action_list   = check_env_space(self.env.action_space)

        if not isinstance(self.input_shape_list, (list, tuple, np.ndarray)):
            self.input_shape_list = [(observation_space[i,0] for i in range(len_obs_list))]

        if not isinstance(self.)
            .................................

    def input_to_hidden(self, i):

        input_layer = Input(self.input_shape_list[i])

        hidden = input_layer
        if len(self.start_indiv_hidden) > 0:
            for j in range(len(self.start_indiv_hidden)):
                hidden = Dense(self.start_indiv_hidden[j, i], activation=self.hidden_activation)(hidden)

        return (input_layer, hidden)


    def hidden_to_output(self, i, output_layer):

        if len(self.end_indiv_hidden) > 0:
            for j in range(len(self.end_indiv_hidden)):
                output_layer = Dense(self.end_indiv_hidden[j, i], activation=self.hidden_activation)(output_layer)

        return Dense(self.outputs_list[i], activation=self.output_activation[i])(output_layer)


    def create_model(self):

        input_layers_and_connect_layers = [self.input_to_hidden(i) for i in range(len(self.input_shape_list))]

        input_layers_and_connect_layers = list(zip(*input_layers_and_connect_layers))
        input_layers   = list(input_layers_and_connect_layers[0])
        connect_layers = list(input_layers_and_connect_layers[1])

        combined = Concatenate(axis=-1)(connect_layers)

        for num_neurons in self.combined_hidden:
            combined = Dense(shape=num_outputs, self.actor_output_act)(combined)

        output_layers = [self.hidden_to_output(i,combined) for i in range(len(self.outputs_list))]

        model = keras.Model(inputs=input_layers, outputs=output_layers)
        return input_layers, output_layers, model


