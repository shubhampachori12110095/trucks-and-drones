import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


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
            agent_index,
            logger,
            max_steps_per_episode,
            use_target_model,
            greed_eps,
            greed_eps_decay,
            greed_eps_min,
            alpha,
            gamma,
            tau,
            loss_function,
            optimizer,
            n_hidden,
            n_actions,
            activation,
    ):

        self.agent_index = agent_index
        self.logger = logger
        self.max_steps_per_episode = max_steps_per_episode
        self.use_target_model = use_target_model

        self.greed_eps = greed_eps
        self.greed_eps_decay = greed_eps_decay
        self.greed_eps_min = greed_eps_min

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.loss_function = loss_function
        self.optimizer = optimizer

        self.model = DQNModel(n_hidden, n_actions, activation)
        if self.use_target_model:
            self.target_model = tf.keras.models.clone_model(self.model)


    def start_gradient_recording(self):

        self.tape = tf.GradientTape()
        self.tape.watch(self.model.trainable_variables)

        self.targets = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.Q_futures = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)

    def pred_outputs(self, inputs):

        if isinstance(inputs, list):
            return [self.model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in inputs])]
        return [self.model(tf.expand_dims(tf.convert_to_tensor(inputs), 0))]


    def act(self, actions, t):
        actions = tf.squeeze(actions[0])

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


    def calc_loss(self):

        self.targets = self.targets.stack()
        self.rewards = self.rewards.stack()
        self.Q_futures = self.Q_futures.stack()

        self.logger.log_mean('Q_future_' + str(self.agent_index), np.mean(tf.squeeze(self.Q_futures).numpy()))
        self.logger.log_mean('reward_'+str(self.agent_index), np.mean(self.rewards.numpy()))

        loss =  self.alpha * self.loss_function(self.targets, (self.rewards + tf.squeeze(self.Q_futures) * self.gamma))
        self.logger.log_mean('loss_' + str(self.agent_index), loss.numpy())

        return loss

    def calc_loss_and_update_weights(self):
        q_loss = self.calc_loss()
        q_gradient = self.tape.gradient(q_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(q_gradient, self.model.trainable_variables))
        return q_loss

    def update_target_weights(self):
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
            max_steps_per_episode,
            use_target_model,
            greed_eps,
            greed_eps_decay,
            greed_eps_min,
            alpha,
            gamma,
            tau,
            loss_function,
            optimizer,
            n_hidden,
            n_actions,
            activation
    ):

        super().__init__(
            0, logger, max_steps_per_episode, use_target_model, greed_eps, greed_eps_decay, greed_eps_min, alpha,
            gamma, tau, loss_function, optimizer, n_hidden, n_actions, activation
        )

        self.env = env

    def train_agent(self, num_episodes):

        for e in range(num_episodes):

            state = self.env.reset()

            self.start_gradient_recording()

            for t in range(self.max_steps_per_episode):

                outputs = self.pred_outputs(state)

                action = self.act(outputs)
                state, reward, done, _ = self.env.step(action.numpy())

                self.reward(reward)
                self.pred_q_future(state, t, done)

                if tf.cast(done, tf.bool):
                    break

            self.calc_loss_and_update_weights()

            if self.use_target_model:
                if e % self.target_update == 0:
                    self.update_target_weights()

            self.logger.print_status(self.env.count_episodes)
