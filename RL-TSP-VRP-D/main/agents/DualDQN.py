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


class DualDQNCore:

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
        self.optimizer_1 = optimizer
        self.optimizer_2 = optimizer

        self.model_1 = DQNModel(n_hidden, n_actions, activation)
        self.model_2 = DQNModel(n_hidden, n_actions, activation)

        if self.use_target_model:
            self.target_model_1 = tf.keras.models.clone_model(self.model_1)
            self.target_model_2 = tf.keras.models.clone_model(self.model_2)


    def start_gradient_recording(self):

        self.tape_1 = tf.GradientTape()
        self.tape_1.watch(self.model_1.trainable_variables)

        self.tape_2 = tf.GradientTape()
        self.tape_2.watch(self.model_2.trainable_variables)

        self.targets_1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.targets_2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.Q_futures_1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.Q_futures_2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)

    def pred_outputs(self, inputs):

        if isinstance(inputs, list):
            outputs_1 = self.model_1([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in inputs])
            outputs_2 = self.model_2([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in inputs])

        else:
            outputs_1 = self.model_1(tf.expand_dims(tf.convert_to_tensor(inputs), 0))
            outputs_2 = self.model_2(tf.expand_dims(tf.convert_to_tensor(inputs), 0))

        return [outputs_1, outputs_2]

    def act(self, actions, t):
        actions_1 = tf.squeeze(actions[0])
        actions_2 = tf.squeeze(actions[1])

        self.greed_eps *= self.greed_eps_decay
        self.greed_eps = max(self.greed_eps, self.greed_eps_min)
        self.logger.log_mean('greed_eps_'+str(self.agent_index), self.greed_eps)

        if np.random.random() < self.greed_eps:
            action = tf.convert_to_tensor(np.random.choice(len(actions)))

        else:
            if tf.math.reduce_max(actions_1) < tf.math.reduce_max(actions_2):
                action = tf.math.argmax(actions_1)
            else:
                action = tf.math.argmax(actions_2)

        self.targets_1 = self.targets_1.write(t, actions_1[action])
        self.targets_2 = self.targets_2.write(t, actions_2[action])

        return action

    def reward(self, rew, t):

        self.rewards = self.rewards.write(t, rew)

    def q_future(self, state, t, done):

        if tf.cast(done, tf.bool):
            q_fut_1 = 0
            q_fut_2 = 0

        else:

            if isinstance(state, list):
                if tf.cast(self.use_target_model, tf.bool):
                    outputs_1 = self.target_model_1([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])
                    outputs_2 = self.target_model_2([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])
                else:
                    outputs_1 = self.model_1([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])
                    outputs_2 = self.model_2([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])

            else:
                if tf.cast(self.use_target_model, tf.bool):
                    outputs_1 = self.target_model_1(tf.expand_dims(tf.convert_to_tensor(state), 0))
                    outputs_2 = self.target_model_2(tf.expand_dims(tf.convert_to_tensor(state), 0))
                else:
                    outputs_1 = self.model_1(tf.expand_dims(tf.convert_to_tensor(state), 0))
                    outputs_2 = self.model_2(tf.expand_dims(tf.convert_to_tensor(state), 0))

            q_fut_1 = tf.math.reduce_max(tf.squeeze(outputs_1))
            q_fut_2 = tf.math.reduce_max(tf.squeeze(outputs_2))

        self.Q_futures_1 = self.Q_futures_1.write(t, tf.cast(tf.squeeze(q_fut_1), dtype=tf.float32))
        self.Q_futures_2 = self.Q_futures_2.write(t, tf.cast(tf.squeeze(q_fut_2), dtype=tf.float32))

    def calc_loss(self):

        self.targets_1 = self.targets_1.stack()
        self.targets_2 = self.targets_2.stack()
        self.rewards = self.rewards.stack()
        self.Q_futures_1 = self.Q_futures_1.stack()
        self.Q_futures_2 = self.Q_futures_2.stack()

        self.logger.log_mean('Q_future_' + str(self.agent_index)+'_1', np.mean(tf.squeeze(self.Q_futures_1).numpy()))
        self.logger.log_mean('Q_future_' + str(self.agent_index)+'_2', np.mean(tf.squeeze(self.Q_futures_2).numpy()))
        self.logger.log_mean('reward_'+str(self.agent_index), np.mean(self.rewards.numpy()))

        loss_1 = self.loss_function(self.targets_1, (self.rewards + tf.squeeze(self.Q_futures_1) * self.gamma))
        loss_2 = self.loss_function(self.targets_2, (self.rewards + tf.squeeze(self.Q_futures_2) * self.gamma))

        self.logger.log_mean('loss_' + str(self.agent_index)+'_1', loss_1.numpy())
        self.logger.log_mean('loss_' + str(self.agent_index)+'_2', loss_2.numpy())

        return loss_1, loss_2

    def calc_loss_and_update_weights(self):
        loss_1, loss_2 = self.calc_loss()
        grad_1 = self.tape_1.gradient(loss_1, self.model_1.trainable_variables)
        grad_2 = self.tape_2.gradient(loss_2, self.model_2.trainable_variables)
        self.optimizer_1.apply_gradients(zip(grad_1, self.model_2.trainable_variables))
        self.optimizer_2.apply_gradients(zip(grad_2, self.model_2.trainable_variables))
        return self.alpha * (loss_1 + loss_2)

    def update_target_weights(self):
        weights = self.model_1.get_weights()
        target_weights = self.target_model_1.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model_1.set_weights(target_weights)

        weights = self.model_2.get_weights()
        target_weights = self.target_model_2.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model_2.set_weights(target_weights)


class DualDQNAgent(DualDQNCore):

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

            outputs = self.pred_outputs(state)

            for t in range(self.max_steps_per_episode):

                action = self.act(outputs)
                state, reward, done, _ = self.env.step(action.numpy())
                self.reward(reward)

                if tf.cast(self.use_target_model, tf.bool):
                    self.q_future(state, t, done)

                if tf.cast(done, tf.bool):
                    break

            self.calc_loss_and_update_weights()

            if self.use_target_model:
                if e % self.target_update == 0:
                    self.update_target_weights()

            self.logger.print_status(self.env.count_episodes)
