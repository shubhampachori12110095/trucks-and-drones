'''

'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

from main.agents.agents import BaseAgent


class BaseAgentBuilder:

    def __init__(self, env, name):

        self.env = env
        self.name = name

        if hasattr(self.env.action_space, '__iter__'):
            self.outputs_list = [[] for i in range(self.env.action_space)]
            self.act_list = [elem.shape for elem in range(self.env.action_space)]
        else:
           self.outputs_list = [[]]
           self.inputs_list = [self.env.action_space.shape]

        if hasattr(self.env.observation_space, '__iter__'):
            self.inputs_list = [elem.shape for elem in range(self.env.observation_space)]
        else:
            self.inputs_list = [self.env.observation_space.shape]

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
            greed_eps_decay: (None, float, int) = 0.999,
            greed_eps_min: (None, float, int) = 0,
            grad_eps: (None, float, int) = 1,
            grad_eps_decay: (None, float, int) = 0.9999,
            grad_eps_min: (None, float, int) = 0.01,
            run_factor: (None, float, int) = None,
            entropy: (None, float, int) = None,
            gamma_q: (None, float, int) = None,
            gamme_v: (None, float, int) = None,
            ):

        self.action_outputs.insert(act_index,{
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
            'gamma_v': gamme_v,
            })

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

    def compile_agents(self):

        self.agents = []
        self.outputs_list = []
        self.activations = []
        actor_index = 0

        for action in self.action_outputs:

            if action['agent'] == 'dqn':
                # DQN
                self.agents.append(DQNAgent(action, actor_index))

                if isinstance(self.acts_list[actor_index], spaces.Discrete):
                    self.outputs_list.append([[self.acts_list[actor_index].shape]])
                    self.activations.append([[action['activation_q']]])
                else:
                    raise Exception('''
                                    Action with index {} was assigned to a DQN, but is not discrete.
                                    '''.format(actor_index)
                                    )

            if action['agent'] == 'sac':

                if isinstance(self.acts_list[actor_index], spaces.Discrete):
                    self.agents.append(DiscreteSACAgent(action, actor_index))

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

                elif isinstance(self.acts_list[actor_index], spaces.Box):
                    self.agents.append(ContinSACAgent(action, actor_index))

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

                else:
                    raise Exception('''
                                    Action with index {} was assigned to a SAC, but is not discrete or contin.
                                    '''.format(actor_index)
                                        )

            if action['agent'] == 'a2c':

                if isinstance(self.acts_list[actor_index], spaces.Discrete):

                    self.agents.append(DiscreteA2CAgent(action, actor_index))

                    self.outputs_list.append([
                        [self.acts_list[actor_index].shape],
                        [1]
                    ])

                    self.activations.append([
                        [action['activation_grad']],
                        [action['activation_val']],
                    ])

                elif isinstance(self.acts_list[actor_index], spaces.Box):

                    self.agents.append(ContinA2CAgent(action, actor_index))

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

                else:
                    raise Exception('''
                                    Action with index {} was assigned to a A2C, but is not discrete or contin.
                                    '''.format(actor_index)
                                    )

            actor_index += 1

        for sup_output in self.supervised_outputs:

            self.agents.append(SupervisedOutputs(sup_output))
            self.outputs_list.append([[sup_output['num_outputs']])
            self.activations.append([[sup_output['activation']]])

    def seperate_input_layer(self):

        self.input_layers = []
        for num_inputs in self.inputs_list:
            self.input_layers.append(layers.Dense(num_inputs, activation="relu"))
        self.all_layers = [self.input_layers]

    def add_hidden_to_each_input(self, num_neurons_factor=2):

        new_layers = []
        for prev_layer in self.all_layers[-1]:
            new_layers.append(layers.Dense(int(prev_layer.shape * num_neurons_factor), activation="relu")(prev_layer))
        self.all_layers.append(new_layers)

    def add_combined_layer(self, num_neurons_factor=1.5):
        self.all_layers.append(layers.Concatenate(axis=-1)(self.all_layers[-1]))
        self.all_layers.append(layers.Dense(
            int(self.all_layers[-1].shape * num_neurons_factor), activation="relu")(self.all_layers[-1]))

    def add_sum_output_neurons_layer(self, num_neurons_factor=0.5):
        self.all_layers.append(layers.Dense(
            int(np.sum(self.outputs_list) * num_neurons_factor), activation="relu")(self.all_layers[-1]))

    def add_output_networks(self, num_layers, num_neurons_factor=1.5):

        self.output_layers = []
        for i in range(self.outputs_list):

            outputs = self.outputs_list[i]
            sum_neurons = sum(outputs)
            net_input = layers.Dense(
                int(num_layers * num_neurons_factor * sum_neurons), activation="relu")(self.all_layers[-1])

            hidden = net_input
            if num_layers > 1:
                for i in range(num_layers - 2):
                    hidden = layers.Dense(
                        int((num_layers - i - 1) * num_neurons_factor * sum_neurons), activation="relu")(hidden)

            for j in range(outputs):
                self.output_layers.append(layer.Dense(outputs[j], activation=self.activations[i][j]))

    def build(self, Agent: BaseAgent = Basegent):
        return Agent(self.name, self.env, self.agents, keras.Model(self.input_layers, self.output_layers))


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







