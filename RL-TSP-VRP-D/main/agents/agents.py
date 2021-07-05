import tensorflow as tf
import numpy as np
'''

'''
class DummyAgent:

    def __init__(self, env, name='DummyAgent', max_steps_per_episode=1000):

        self.name = name
        self.env = env
        self.max_steps_per_episode = max_steps_per_episode

    def train_agent(self, num_episodes):

        actions = []

        for e in range(num_episodes):

            state = self.env.reset()
            done = False

            for i in range(self.max_steps_per_episode):
                state, reward, done, _ = self.env.step(actions)

                if done:
                    break

    def test_agent(self, num_episodes):

        actions = []

        for e in range(num_episodes):

            state = self.env.reset()
            done = False

            for i in range(self.max_steps_per_episode):
                self.env.render()
                state, reward, done, _ = self.env.step(actions)

                if done:
                    break


class BaseAgent:

    def __init__(self, name, env, agents, model, optimizer, tau, use_target_model=True, target_update=1, max_steps_per_episode=1000):

        self.name = name
        self.env = env
        self.agents = agents
        self.model = model
        self.optimizer = optimizer
        self.tau = tau
        self.use_target_model = use_target_model
        self.target_update = target_update

        self.save_interval = 100
        self.save_path = '/models'

        if self.use_target_model:
            self.target_model = model.copy()

        self.max_steps_per_episode = max_steps_per_episode

        self.out_ind = [agent.indices() for agent in self.agents]
        self.q_ind = [agent.agent_index for agent in self.agents if agent.agent_type == 'dqn' or agent.agent_type == 'sac']
        self.q_out_ind = [self.agents[i].indices() for i in self.q_ind]

    @tf.function
    def train_agent(self, num_episodes):

        for e in range(num_episodes):

            state = self.env.reset()

            with tf.GradientTape() as tape:

                for t in tf.range(self.max_steps_per_episode):
                    outputs_list = [self.model(tf.expand_dims(state,0))]
                    actions_list = [self.agents[i].act(outputs_list[self.out_ind[i]], t) for i in range(len(self.agents))]
                    state, rewards, done, _ = self.tf_env_step(actions_list)
                    [self.agents[i].reward(rewards[i], t) for i in range(len(self.agents))]

                    if tf.cast(self.use_target_model, tf.bool):
                        if tf.cast(done, tf.bool):
                            [self.agents[i].q_future(0) for i in self.q_ind]
                        else:
                            outputs_list = [self.target_model(tf.expand_dims(state, 0))]
                            [self.agents[i].q_future(outputs_list[self.q_out_ind[i]]) for i in self.q_ind]

                    if tf.cast(done, tf.bool):
                        break

                loss = sum([agent.calc_loss() for agent in self.agents])

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.use_target_model:
                if e % self.target_update == 0:
                    weights = self.model.get_weights()
                    target_weights = self.target_model.get_weights()
                    for i in range(len(target_weights)):
                        target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
                    self.target_model.set_weights(target_weights)

            if e == num_episodes - 1 or e % self.save_interval == 0:
                self.model.save(self.save_path)


    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action: List[tf.Tensor]) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action],
                                 [tf.float32, tf.int32, tf.int32])



class DQNAgent:

    def __init__(self, params)

        self.agent_type = 'dqn'
        self.agent_index = params['act_index']
        self.optimizer_q = params['optimizer_q']
        self.activation_q = params['activation_q']
        self.loss_q = params['loss_q']
        self.greed_eps = params['greed_eps']
        self.greed_eps_decay = params['greed_eps_decay']
        self.greed_eps_min = params['greed_eps_min']
        self.gamma_q = params['gamma_q']
        self.alpha = params['alpha_q']

        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.targets = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True)
        self.Q_futures = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True)

    def act(self, actions, t):
        self.greed_eps *= self.greed_eps_decay
        self.greed_eps = max(self.greed_eps, self.greed_eps_min)

        if np.random.random() < self.greed_eps:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(np.squeeze(actions))

        self.targets = self.targets.write(t, actions[action])

        return action

    def reward(self, rew):
        self.rewards = self.rewards.write(t, rew)

    def q_future(self, q_fut):
        self.Q_futures = self.Q_futures.write(t, q_fut)

    def calc_loss(self):
        self.targets = self.targets.stack()
        self.rewards = self.rewards.stack()
        self.Q_futures = self.Q_futures.stack()
        return self.alpha * self.loss_function(targets, (rewards + Q_futures * self.gamma_q))















