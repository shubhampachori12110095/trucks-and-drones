'''

'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from gym import spaces

from main.agents.multi_agent import MultiAgent
from main.logger import TrainingLogger, StatusPrinter


class BaseCommonNetwork:

    def __init__(self, inputs_list):

        self.inputs_list = inputs_list
        self.input_layers = []
        self.combined = None

    def seperate_input_layer(self, activation="relu"):

        first_hidden = []

        for input_shape in self.inputs_list:

            self.input_layers.append(layers.Input(shape=input_shape))
            flattened = layers.Flatten()(self.input_layers[-1])
            first_hidden.append(layers.Dense(int(flattened.shape.as_list()[-1]), activation=activation)(flattened))

        self.last_layers_list = first_hidden

    def combined_input_layer(self, activation="relu"):

        self.input_layers.append(layers.Input(shape=self.inputs_list[0]))
        flattened = layers.Flatten()(self.input_layers[0])
        self.combined = layers.Dense(int(flattened.shape.as_list()[-1]), activation=activation)(flattened)

    def add_hidden_to_each_input(self, num_neurons_factor=2, activation="relu"):

        new_layers = []
        for prev_layer in self.last_layers_list:

            new_layers.append(
                layers.Dense(
                    int(prev_layer.shape.as_list()[-1] * num_neurons_factor), activation=activation
                )(prev_layer)
            )

        self.last_layers_list = new_layers

    def combine_layers(self):
        print(self.last_layers_list)
        print(len(self.last_layers_list))
        if len(self.last_layers_list) > 1:
            print('1test')
            self.combined = layers.Concatenate(axis=-1)(self.last_layers_list)
            print(self.combined)
        else:
            print(self.last_layers_list[0])
            self.combinded = self.last_layers_list[0]
            print(self.last_layers_list[0])
            print(self.combined)
            print('test2')
        print(self.combined)

    def add_combined_layer(self, num_neurons_factor=1.5, activation="relu"):

        print(self.combined)
        self.combined = layers.Dense(
            int(self.combined.shape.as_list()[-1] * num_neurons_factor), activation=activation
        )(self.combined)

    def build_model(self):
        '''
        if len(self.input_layers) > 1:
            self.seperate_input_layer()

            if self.combined is None:
                self.combine_layers()
                self.add_combined_layer()
                self.add_combined_layer()

        else:
            self.combined_input_layer()
            #self.add_combined_layer(6)
            #self.add_combined_layer(6)
        '''
        self.input_layers.append(layers.Input(shape=self.inputs_list[0]))
        self.combined = layers.Flatten()(self.input_layers[0])
        return keras.Model(self.input_layers, self.combined)



class BaseAgentBuilder:

    def __init__(self, env, log_dir=None, CommonNetwork=None, name='test'):

        self.env = env
        try:
            self.name = env.name
        except:
            self.name = name

        if log_dir is None:
            self.logger = StatusPrinter(self.name)
        else:
            self.logger = TrainingLogger(self.name, log_dir)

        if isinstance(self.env.action_space, spaces.Tuple):
            self.action_outputs = [[] for i in range(len(self.env.action_space))]
            self.acts_list = [elem for elem in self.env.action_space]
        else:
           self.action_outputs = [[]]
           self.acts_list = [self.env.action_space]

        if isinstance(self.env.observation_space, spaces.Tuple):
            self.inputs_list = [elem.shape for elem in self.env.observation_space]
        else:
            self.inputs_list = [self.env.observation_space.shape]

        print('self.inputs_list',self.inputs_list)
        print('self.inputs_list',self.inputs_list)

        self.supervised_outputs = []

        if CommonNetwork == None:
            self.common_model = BaseCommonNetwork(self.inputs_list)
        else:
            self.common_model = CommonNetwork(self.inputs_list)

        self.agents = [None for elem in self.action_outputs]
        self.agents_with_target_models = []
        self.agents_with_q_future = []

    def assign_agent(self, agent, at_index):

        agent.finish_init(at_index, self.logger, self.acts_list[at_index])

        self.agents[at_index] = agent

        if agent.use_target_model:
            self.agents_with_target_models.append(at_index)

        if agent.uses_q_future:
            self.agents_with_q_future.append(at_index)

    def add_supervised_outputs(
            self,
            num_outputs,
            optimizer,
            activation,
            loss,
        ):

        self.supervised_outputs.append({
            'num_outputs': num_outputs,
            'optimizer': optimizer,
            'activation': activation,
            'loss': loss,
        })


    def check_actions_for_assignment(self, dont_break=True):

        sth_not_assigned = False
        for i in range(len(self.agents)):
            if self.agents[i] is None:
                print('Action with index {} was not assigned to an actor.'.format(i))
                sth_not_assigned = True

        if not dont_break and sth_not_assigned:
            raise Exception('All actions must be assigned to an actor.')


    def build(
            self,
            Agent: MultiAgent = MultiAgent,
            optimizer: optimizer_v2.OptimizerV2 = keras.optimizers.Adam(),
            tau: float = 0.15,
            target_update: (None, int) = 1,
            max_steps_per_episode = 1000,
            actions_as_list = False
        ):

        self.check_actions_for_assignment(dont_break=False)

        self.common_model = self.common_model.build_model()
        self.common_model.summary()

        if target_update is None:
            use_target_model = False
        else:
            use_target_model = True

        return Agent(self.name, self.env, self.agents, self.common_model, self.logger,
                     self.agents_with_target_models, self.agents_with_q_future,
                     optimizer, tau, use_target_model, target_update, max_steps_per_episode, actions_as_list
                     )

