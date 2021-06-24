import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class Agents:

    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.action_outputs = []

    def add_action(
            self,
            space: str = 'discrete',  # 'discrete', 'contin'
            optimizer_q: (None, optimizer_v2.OptimizerV2) = None,
            optimizer_grad: (None, optimizer_v2.OptimizerV2) = None,
            optimizer_val: (None, optimizer_v2.OptimizerV2) = None,
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

        self.action_outputs.append({
            'space': space,
            'optimizer_q': optimizer_q,
            'optimizer_grad': optimizer_grad,
            'optimizer_val': optimizer_val,
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

    def compile(
            self,
            tau: (None, float, int) = None,
            update_target: int = 100,
            ):

        for action in self.action_outputs:

            if not action['optimizer_q'] is None: # True

                if action['optimizer_grad'] is None and action['optimizer_val'] is None:
                    # DQN
                    agent = CoreAgent('dqn')

                elif not action['optimizer_grad'] is None and not action['optimizer_val'] is None:
                    # SAC
                    agent = CoreAgent('sac')
                else:
                    raise Exception("""
                                    'Either optimizer_grad' was {} and 'optimizer_val' was {}, but both must be None or not None,
                                    when 'optimizer_q' is not None.
                                    """.format(action['optimizer_grad'], action['optimizer_val']))

            else:

                if action['optimizer_grad'] is None or action['optimizer_val'] is None:
                    raise Exception("""
                                    'optimizer_grad' was {} or 'optimizer_val' was {}, but can't be None, when 'optimizer_q' is None.
                                    """.format(action['optimizer_grad'],action['optimizer_val']))
                else:
                    #A2C
                    agent = CoreAgent('a2c')


class CoreAgent:

    def __init__(self, agent_type):
        self.agent_type = agent_type

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






class DummyAgent:

    def __init__(self, env, name='DummyAgent'):

        self.name = name
        self.env = env

    def train_agent(self, num_episodes):

        actions = []

        for e in range(num_episodes):

            state = self.env.reset()
            done = False

            while not done:
                state, reward, done, _ = self.env.step(actions)

    def test_agent(self, num_episodes):

        actions = []

        for e in range(num_episodes):

            state = self.env.reset()
            done = False

            while not done:
                self.env.render()
                state, reward, done, _ = self.env.step(actions)
