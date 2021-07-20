'''

'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from gym import spaces

from main.agents.agents import BaseAgent, DQNAgent, DiscreteA2CAgent
from main.logger import TrainingLogger


class BaseCommonNetwork:

    def __init__(self, inputs_list):

        self.inputs_list = inputs_list

    def seperate_input_layer(self, activation="relu"):

        first_hidden = []
        self.input_layers = []

        for input_shape in self.inputs_list:

            self.input_layers.append(layers.Input(shape=input_shape))
            flattened = layers.Flatten()(self.input_layers[-1])
            first_hidden.append(layers.Dense(int(flattened.shape.as_list()[-1]), activation=activation)(flattened))

        self.last_layers_list = first_hidden

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

        self.combined = layers.Concatenate(axis=-1)(self.last_layers_list)

    def add_combined_layer(self, num_neurons_factor=1.5, activation="relu"):

        self.combined = layers.Dense(
            int(self.combined.shape.as_list()[-1] * num_neurons_factor), activation=activation
        )(self.combined)

    def build(self):
        return keras.Model(self.input_layers, self.combined)



class BaseAgentBuilder:

    def __init__(self, env, log_dir=None, CommonNetwork=BaseCommonNetwork):

        self.env = env
        self.name = env.name

        if log_dir is None:
            self.logger = DummyLogger(self.name)
        else:
            self.logger = TrainingLogger(self.name, log_dir)

        if isinstance(self.env.action_space, spaces.Tuple):
            self.action_outputs = [[] for i in range(len(self.env.action_space))]
            self.acts_list = [elem for elem in self.env.action_space]
        else:
           self.action_outputs = [[]]
           self.inputs_list = [self.env.action_space]

        if isinstance(self.env.observation_space, spaces.Tuple):
            self.inputs_list = [elem.shape for elem in self.env.observation_space]
        else:
            self.inputs_list = [self.env.observation_space.shape]

        print('self.inputs_list',self.inputs_list)

        self.supervised_outputs = []

        self.common_model = CommonNetwork(self.inputs_list)

    def assign_agent_to_act(
            self,
            agent: str = 'dqn',  # 'dqn', 'a2c', 'sac'
            act_index: int = 0,
            optimizer_q: (None, optimizer_v2.OptimizerV2) = None,
            optimizer_grad: (None, optimizer_v2.OptimizerV2) = None,
            optimizer_val: (None, optimizer_v2.OptimizerV2) = None,
            activation_q: (None, str) = None,
            activation_grad: (None, str, list, tuple, np.ndarray) = None,
            activation_val: (None, str) = None,
            loss_q: (None, str) = 'mse',  # 'mse'
            loss_grad: (None, str) = 'huber',  # 'huber'
            loss_val: (None, str) = 'mse',  # 'mse'
            greed_eps: (None, float, int) = 1,
            greed_eps_decay: (None, float, int) = 0.99999,
            greed_eps_min: (None, float, int) = 0.05,
            grad_eps: (None, float, int) = 1,
            grad_eps_decay: (None, float, int) = 0.9999,
            grad_eps_min: (None, float, int) = 0.01,
            run_factor: (None, float, int) = None,
            entropy: (None, float, int) = None,
            gamma_q: (None, float, int) = None,
            gamma_v: (None, float, int) = None,
            alpha_grad: (None, float, int) = 0.01,
            alpha_val: (None, float, int) = 0.01,
            alpha_q: (None, float, int) = 0.01,
            ):

        self.action_outputs[act_index] = {
            'agent': agent,
            'act_index': act_index,
            'optimizer_q': optimizer_q,
            'optimizer_grad': optimizer_grad,
            'optimizer_val': optimizer_val,
            'activation_q': activation_q,
            'activation_grad': activation_grad,
            'activation_val': activation_val,
            'loss_q': loss_q,
            'loss_grad': loss_grad,
            'loss_val': loss_val,
            'greed_eps': greed_eps,
            'greed_eps_decay': greed_eps_decay,
            'greed_eps_min': greed_eps_min,
            'grad_eps': grad_eps,
            'grad_eps_decay': grad_eps_decay,
            'grad_eps_min': grad_eps_min,
            'run_factor': run_factor,
            'entropy': entropy,
            'gamma_q': gamma_q,
            'gamma_v': gamma_v,
            'alpha_grad': alpha_grad,
            'alpha_val': alpha_val,
            'alpha_q': alpha_q,
        }

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

    def assign_out_indices(self, num_outs, cur_count, uses_target=False):
        self.out_ind.append([cur_count+i for i in range(num_outs)])
        cur_count += len(self.out_ind[-1])
        if uses_target:
            self.q_out_ind.append(self.out_ind[-1])
        return cur_count

    def compile_agents(self):

        self.agents = []
        self.outputs_list = []
        self.activations = []
        actor_index = 0

        self.out_ind = []
        self.q_out_ind = []
        indice_count = 0

        for action in self.action_outputs:

            if isinstance(action, list):
                raise Exception('Action with index {} was not assigned to an actor.'.format(actor_index))

            if action['agent'] == 'dqn':
                # DQN
                self.agents.append(DQNAgent(action, self.logger))

                if isinstance(self.acts_list[actor_index], spaces.Discrete):
                    self.outputs_list.append([[self.acts_list[actor_index].n]])
                    self.activations.append([[action['activation_q']]])
                    indice_count = self.assign_out_indices(1,indice_count, True)
                    self.agents[-1].num_actions = int(self.acts_list[actor_index].n)
                else:
                    raise Exception('''
                                    Action with index {} was assigned to a DQN, but is not discrete.
                                    '''.format(actor_index)
                                    )

            if action['agent'] == 'sac':

                if isinstance(self.acts_list[actor_index], spaces.Discrete):
                    self.agents.append(DiscreteSACAgent(action, self.logger))

                    self.outputs_list.append([
                        [self.acts_list[actor_index].shape],
                        [self.acts_list[actor_index].shape],
                        [1]
                    ])

                    self.activations.append([
                        [action['activation_q']],
                        [action['activation_grad']],
                        [action['activation_val']],
                    ])

                    indice_count = self.assign_out_indices(3, indice_count, True)

                elif isinstance(self.acts_list[actor_index], spaces.Box):
                    self.agents.append(ContinSACAgent(action, self.logger))

                    self.outputs_list.append([
                        [self.acts_list[actor_index].shape],
                        [self.acts_list[actor_index].shape],
                        [self.acts_list[actor_index].shape],
                        [1]
                    ])

                    self.activations.append([
                        [action['activation_q']],
                        [action['activation_grad'][0]],
                        [action['activation_grad'][1]],
                        [action['activation_val']],
                    ])

                    indice_count = self.assign_out_indices(4, indice_count, True)

                else:
                    raise Exception('''
                                    Action with index {} was assigned to a SAC, but is not discrete or contin.
                                    '''.format(actor_index)
                                        )

            if action['agent'] == 'a2c':

                if isinstance(self.acts_list[actor_index], spaces.Discrete):

                    self.agents.append(DiscreteA2CAgent(action, self.logger))

                    self.outputs_list.append([
                        [self.acts_list[actor_index].shape],
                        [1]
                    ])

                    self.activations.append([
                        [action['activation_grad']],
                        [action['activation_val']],
                    ])

                    indice_count = self.assign_out_indices(2, indice_count, True)

                elif isinstance(self.acts_list[actor_index], spaces.Box):

                    self.agents.append(ContinA2CAgent(action, self.logger))

                    self.outputs_list.append([
                        [self.acts_list[actor_index].shape],
                        [self.acts_list[actor_index].shape],
                        [1]
                    ])

                    self.activations.append([
                        [action['activation_grad'][0]],
                        [action['activation_grad'][1]],
                        [action['activation_val']],
                    ])

                    indice_count = self.assign_out_indices(3, indice_count, True)

                else:
                    raise Exception('''
                                    Action with index {} was assigned to a A2C, but is not discrete or contin.
                                    '''.format(actor_index)
                                    )

            actor_index += 1

        for sup_output in self.supervised_outputs:

            self.agents.append(SupervisedOutputs(sup_output, self.logger))
            self.outputs_list.append([[sup_output['num_outputs']]])
            self.activations.append([[sup_output['activation']]])



    def add_output_networks(self, num_layers=3, num_neurons_factor=1.5):

        self.output_layers = []
        for i in range(len(self.outputs_list)):

            outputs = self.outputs_list[i]
            for out in outputs:
                sum_neurons = sum(out)
                hidden = layers.Dense(
                    int(num_layers * num_neurons_factor * sum_neurons), activation="relu")(self.combined)

                if num_layers > 1:
                    for i in range(num_layers - 2):
                        hidden = layers.Dense(
                            int((num_layers - i - 1) * num_neurons_factor * sum_neurons), activation="relu")(hidden)

                for j in range(len(out)):
                    if self.activations[i][j][0] is None:
                        self.output_layers.append(layers.Dense(out[j])(hidden))
                    else:
                        self.output_layers.append(layers.Dense(out[j], activation=self.activations[i][j][0])(hidden))

    def build(
            self,
            Agent: BaseAgent = BaseAgent,
            optimizer: optimizer_v2.OptimizerV2 = keras.optimizers.Adam(),
            tau: float = 0.15,
            target_update: (None, int) =1,
            max_steps_per_episode=1000
        ):

        model = keras.Model(self.input_layers, self.output_layers)
        model.summary()

        if target_update is None:
            use_target_model = False
        else:
            use_target_model = True

        return Agent(self.name, self.env, self.agents, model, self.logger, self.out_ind, self.q_out_ind,
                     optimizer, tau, use_target_model, target_update, max_steps_per_episode,
                     )


class CoreAgent:

    def __init__(self, agent_type):
        self.agent_type = agent_type

    def random_act(self):
        return env.action_space.sample()

    def random_auto_act(self):
        return None

    def random_act_by_prob(self):
        return tf.random.categorical(action_logits_t, 1)[0, 0]

    def greedy_epsilon(self, actions):
        self.greed_eps *= self.greed_eps_decay
        self.greed_eps = max(self.greed_eps, self.greed_eps_min)

        if np.random.random() < self.greed_eps:
            if self.greedy_gradient():
                #return tf.random.categorical(action_logits_t, 1)[0, 0]
                return np.random.choice(self.num_actions, p=np.squeeze(action_probs))
            else:
                return np.random.choice(self.num_actions)
        else:
            return np.argmax(np.squeeze(action_probs))

    def logits_to_action(self, action_logits_t):
        # Sample next action from the action probability distribution

        return tf.nn.softmax(action_logits_t)







