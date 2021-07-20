import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from gym import spaces


class DQNModel(tf.keras.Model):

    def __init__(
            self,
            n_hidden: int,
            n_actions: int,
            activation: str
    ):

        super().__init__()

        self.flat = layers.Flatten()
        self.common = layers.Dense(n_hidden, activation=activation)
        self.q_layer = layers.Dense(n_actions)

    def call(self, inputs: tf.Tensor):

        if isinstance(inputs, list):
            inputs = self.flat(tf.stack(inputs))

        x = self.common(inputs)
        return self.q_layer(x)


class DQNCore:

    def __init__(
            self,
            use_target_model: bool = True,
            greed_eps: float = 1.0,
            greed_eps_decay: float = 0.99999,
            greed_eps_min: float = 0.05,
            alpha: float = 0.1,
            gamma: float = 0.9,
            tau: float = 0.15,
            loss_function: tf.keras.losses.Loss = tf.keras.losses.MeanSquaredError(),
            optimizer: optimizer_v2.OptimizerV2 = tf.keras.optimizers.Adam(),
            n_hidden: int = 64,
            activation: str = 'relu',
    ):

        self.uses_q_future = True
        self.use_target_model = use_target_model

        self.greed_eps = greed_eps
        self.greed_eps_decay = greed_eps_decay
        self.greed_eps_min = greed_eps_min

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.loss_function = loss_function
        self.optimizer = optimizer

        self.n_hidden = n_hidden
        self.activation = activation

    def finish_init(self, agent_index, logger, n_actions):
        self.logger = logger
        self.agent_index = agent_index

        if isinstance(n_actions, spaces.Box):
            raise Exception(
                'DQN with index {} was assigned to continuous actions! Change to spaces.Discrete().'.format(self.agent_index))

        elif isinstance(n_actions, spaces.Discrete):
            n_actions = n_actions.n

        elif not isinstance(n_actions, int):
            raise Exception(
                'DQN with index {} was assigned to {} which can not be interpreted! Change to spaces.Discrete().'.format(self.agent_index, n_actions))

        self.model = DQNModel(self.n_hidden, n_actions, self.activation)

        if self.use_target_model:
            self.target_model = DQNModel(self.n_hidden, n_actions, self.activation)

    def start_gradient_recording(self):

        self.tape = tf.GradientTape(persistent=True)

        self.tape._push_tape()

        self.tape.watch(self.model.trainable_variables)

        self.targets = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.Q_futures = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)

    def act(self, inputs, t):

        if isinstance(inputs, list):
            outputs = self.model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in inputs])
        else:
            outputs = self.model(tf.expand_dims(tf.convert_to_tensor(inputs), 0))

        actions = tf.squeeze(outputs)

        self.greed_eps *= self.greed_eps_decay
        self.greed_eps = max(self.greed_eps, self.greed_eps_min)
        self.logger.log_mean('greed_eps_'+str(self.agent_index), self.greed_eps)

        if np.random.random() < self.greed_eps:
            action = tf.convert_to_tensor(np.random.choice(len(actions)))
        else:
            action = tf.math.argmax(actions)

        self.targets = self.targets.write(t, actions[action])

        return action

    def reward(self, rew, t):

        self.rewards = self.rewards.write(t, rew)

    def pred_q_future(self, state, t, done):

        if tf.cast(done, tf.bool):
            q_fut = 0

        else:

            if isinstance(state, list):

                if tf.cast(self.use_target_model, tf.bool):
                    outputs = self.target_model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])
                else:
                    outputs = self.model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])

            else:

                if tf.cast(self.use_target_model, tf.bool):
                    outputs = self.target_model(tf.expand_dims(tf.convert_to_tensor(state), 0))
                else:
                    outputs = self.model(tf.expand_dims(tf.convert_to_tensor(state), 0))

            q_fut = tf.math.reduce_max(tf.squeeze(outputs))

        self.Q_futures = self.Q_futures.write(t, tf.cast(tf.squeeze(q_fut), dtype=tf.float32))

    def calc_loss_and_update_weights(self):

        self.targets = self.targets.stack()
        self.rewards = self.rewards.stack()
        self.Q_futures = self.Q_futures.stack()

        #print(self.targets)
        #print(self.rewards)
        #print(self.Q_futures)

        self.logger.log_mean('Q_future_' + str(self.agent_index), np.mean(tf.squeeze(self.Q_futures).numpy()))
        self.logger.log_mean('reward_'+str(self.agent_index), np.mean(self.rewards.numpy()))

        loss = self.loss_function(self.targets, (self.rewards + tf.squeeze(self.Q_futures) * self.gamma))
        self.logger.log_mean('loss_' + str(self.agent_index), loss.numpy())

        self.tape._pop_tape()
        grads = self.tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        del self.tape

        return self.alpha * loss

    def update_target_weights(self):
        if self.use_target_model:
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
            self.target_model.set_weights(target_weights)


class DQNAgent(DQNCore):

    def __init__(
            self,
            env,
            logger,
            n_actions: (int, spaces.Space),
            max_steps_per_episode: int,
            use_target_model: bool = True,
            greed_eps: float = 1.0,
            greed_eps_decay: float = 0.999,
            greed_eps_min: float = 0.05,
            alpha: float = 0.1,
            gamma: float = 0.9,
            tau: float = 0.15,
            loss_function: tf.keras.losses.Loss = tf.keras.losses.MeanSquaredError(),
            optimizer: optimizer_v2.OptimizerV2 = tf.keras.optimizers.Adam(),
            n_hidden: int = 128,
            activation: str = 'relu',
    ):

        super().__init__(
            use_target_model, greed_eps, greed_eps_decay, greed_eps_min, alpha,
            gamma, tau, loss_function, optimizer, n_hidden, activation
        )

        self.env = env
        self.finish_init(0, logger, n_actions)
        self.max_steps_per_episode = max_steps_per_episode

    def train_agent(self, num_episodes):

        for e in range(num_episodes):

            state = self.env.reset()

            self.start_gradient_recording()

            for t in range(self.max_steps_per_episode):

                action = self.act(state)
                state, reward, done, _ = self.env.step(action.numpy())

                self.reward(reward)
                self.pred_q_future(state, t, done)

                if tf.cast(done, tf.bool):
                    break

            self.calc_loss_and_update_weights()

            if e % self.target_update == 0:
                self.update_target_weights()

            self.logger.print_status(self.env.count_episodes)
