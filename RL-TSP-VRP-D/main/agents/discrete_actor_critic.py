import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from gym import spaces


class DiscreteActorCriticModel(tf.keras.Model):

    def __init__(
            self,
            n_actions: int,
            activation: str = 'relu',
            common_layers: (None, list) = None,
            critic_layers: (None, list) = None,
            actor_layers: (None, list) = None,
    ):
        super().__init__()

        self.flat = layers.Flatten()

        if common_layers is None:
            self.common = [layers.Dense(int(n_actions*1.5), activation='relu')]
        else:
            self.common = common_layers

        if critic_layers is None:
            self.critic = [
                layers.Dense(int(n_actions*2), activation='relu'),
                layers.Dense(int(n_actions), activation='relu'),
                #layers.Dense(2, activation='tanh'),
                layers.Dense(1)
            ]

        else:
            self.critic = critic_layers

        if actor_layers is None:
            self.actor = [layers.Dense(n_actions)]
        else:
            self.actor = actor_layers


    def call(self, inputs: tf.Tensor):

        if isinstance(inputs, list):
            inputs = self.flat(tf.stack(inputs))

        common_x = inputs
        for elem in self.common: common_x = elem(common_x)

        actor_out = common_x
        for elem in self.actor: actor_out = elem(actor_out)

        critic_out = common_x
        for elem in self.critic: critic_out = elem(critic_out)

        return critic_out, actor_out


class DiscreteActorCriticCore:

    def __init__(
            self,
            greed_eps: float = 1.0,
            greed_eps_decay: float = 0.99999,
            greed_eps_min: float = 0.1,
            alpha_common: float = 1.0,
            alpha_actor: float = 0.5,
            alpha_critic: float = 0.5,
            gamma: float = 0.9,
            standardize: bool = False,
            loss_function_critic: tf.keras.losses.Loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM),
            optimizer_common: optimizer_v2.OptimizerV2 = tf.keras.optimizers.Adam(),
            optimizer_actor: optimizer_v2.OptimizerV2 = tf.keras.optimizers.Adam(learning_rate=0.00001),
            optimizer_critic: optimizer_v2.OptimizerV2 = tf.keras.optimizers.Adam(),
            n_hidden: int = 64,
            activation: str = 'relu',
            global_standardize = True,
    ):

        self.uses_q_future = False
        self.use_target_model = False

        self.greed_eps = greed_eps
        self.greed_eps_decay = greed_eps_decay
        self.greed_eps_min = greed_eps_min

        self.alpha_common = alpha_common
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        self.standardize = standardize
        self.global_standardize = global_standardize

        if self.global_standardize:
            self.global_max = tf.constant(0.0, dtype=tf.float32)

        self.loss_function_critic = loss_function_critic
        self.optimizer_common = optimizer_common
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic

        self.n_hidden = n_hidden
        self.activation = activation

        self.eps = np.finfo(np.float32).eps.item()

    def finish_init(self, agent_index, logger, n_actions):
        self.logger = logger
        self.agent_index = agent_index

        if isinstance(n_actions, spaces.Box):
            raise Exception(
                'Discrete Actor Critic with index {} was assigned to continuous actions! Change to spaces.Discrete().'.format(self.agent_index))

        elif isinstance(n_actions, spaces.Discrete):
            n_actions = n_actions.n

        elif not isinstance(n_actions, int):
            raise Exception(
                'Discrete Actor Critic with index {} was assigned to {} which can not be interpreted! Change to spaces.Discrete().'.format(self.agent_index, n_actions))

        self.model = DiscreteActorCriticModel(n_actions, self.activation)

        if self.use_target_model:
            self.target_model = DiscreteActorCriticModel(n_actions, self.activation)

    def start_gradient_recording(self):

        self.tape_common = tf.GradientTape(persistent=True)
        self.tape_critic = tf.GradientTape(persistent=True)
        self.tape_actor = tf.GradientTape(persistent=True)

        self.tape_common._push_tape()
        self.tape_critic._push_tape()
        self.tape_actor._push_tape()

        self.tape_common.watch([elem.trainable_variables for elem in self.model.common])
        self.tape_critic.watch([elem.trainable_variables for elem in self.model.critic])
        self.tape_actor.watch([elem.trainable_variables for elem in self.model.actor])

        self.values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)

    def act(self, inputs, t, info_dict):

        if isinstance(inputs, list):
            val, actions = self.model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in inputs])
        else:
            val, actions = self.model(tf.expand_dims(tf.convert_to_tensor(inputs), 0))

        action = tf.random.categorical(actions[0], 1)[0, 0]

        self.values = self.values.write(t, tf.squeeze(val))
        self.act_probs = self.act_probs.write(t, tf.squeeze(tf.nn.softmax(actions[0])[0, action]))

        return action

    def reward(self, rew, t):

        self.rewards = self.rewards.write(t, rew)

    def get_expected_return(self):

        sample_len = tf.shape(self.rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=sample_len)

        rewards = tf.cast(self.rewards[::-1], dtype=tf.float32)#/tf.cast(sample_len, dtype=tf.float32)

        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(sample_len):
            discounted_sum = rewards[i] + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]


        if self.standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + self.eps))

        if self.global_standardize:
            returns = self.calc_global_norm(returns)

        return returns

    def calc_global_norm(self, returns):
        max_abs_return = tf.math.reduce_mean(returns)
        self.global_max = tf.math.reduce_mean(tf.stack([self.global_max, max_abs_return]))
        return ((returns - (self.global_max/2)) /
                       (tf.math.reduce_std(returns) + self.eps))

    def loss_function_actor(self, act_probs, advantage, loss_critic):

        act_log_probs = tf.math.log(act_probs)
        loss_actor = -tf.math.reduce_sum(act_log_probs * advantage)
        '''
        if tf.reduce_mean(advantage) < 0:
            loss_actor = loss_actor - loss_critic
        else:
            loss_actor = loss_actor + loss_critic
        '''

        return loss_actor

    def calc_loss_and_update_weights(self):

        self.values = tf.squeeze(self.values.stack())
        self.act_probs = tf.squeeze(self.act_probs.stack())
        self.rewards = tf.squeeze(self.rewards.stack())

        #print(self.values)

        returns = self.get_expected_return()
        '''
        print('self.values')
        print(self.values)
        print('self.act_probs')
        print(self.act_probs)
        print('self.rewards')
        print(self.rewards)
        print('returns')
        print(returns)
        '''

        self.logger.log_mean('values_' + str(self.agent_index), np.mean(self.values.numpy()))
        self.logger.log_mean('act_probs_' + str(self.agent_index), np.mean(self.act_probs.numpy()))
        self.logger.log_mean('rewards_'+str(self.agent_index), np.mean(self.rewards.numpy()))
        self.logger.log_mean('returns_'+str(self.agent_index), np.mean(returns.numpy()))

        loss_critic = self.loss_function_critic(tf.expand_dims(self.values, 1), tf.expand_dims(returns, 1))
        loss_actor = self.loss_function_actor(tf.expand_dims(self.act_probs, 1), tf.expand_dims(returns - self.values, 1), loss_critic)
        loss_common = (self.alpha_actor * loss_actor) + (self.alpha_critic * loss_critic)

        self.logger.log_mean('loss_critic_' + str(self.agent_index), loss_critic.numpy())
        self.logger.log_mean('loss_actor_' + str(self.agent_index), loss_actor.numpy())
        self.logger.log_mean('loss_common_' + str(self.agent_index), loss_common.numpy())

        self.tape_critic._pop_tape()
        self.tape_actor._pop_tape()
        self.tape_common._pop_tape()

        grads_critic = self.tape_critic.gradient(loss_critic, [elem.trainable_variables[0] for elem in self.model.critic])
        grads_actor = self.tape_actor.gradient(loss_actor, [elem.trainable_variables[0] for elem in self.model.actor])
        grads_common = self.tape_common.gradient(loss_common, [elem.trainable_variables[0] for elem in self.model.common])

        self.optimizer_critic.apply_gradients(zip(grads_critic, [elem.trainable_variables[0] for elem in self.model.critic]))
        self.optimizer_actor.apply_gradients(zip(grads_actor, [elem.trainable_variables[0] for elem in self.model.actor]))
        self.optimizer_common.apply_gradients(zip(grads_common, [elem.trainable_variables[0] for elem in self.model.common]))

        del self.tape_critic
        del self.tape_actor
        del self.tape_common

        return self.alpha_common * loss_common


class DiscreteActorCriticAgent(DiscreteActorCriticCore):

    def __init__(
            self,
            env,
            logger,
            n_actions: (int, spaces.Space),
            max_steps_per_episode: int,
            greed_eps: float = 1.0,
            greed_eps_decay: float = 0.99999,
            greed_eps_min: float = 0.05,
            alpha_common: float = 0.1,
            alpha_actor: float = 0.5,
            alpha_critic: float = 0.5,
            gamma: float = 0.9,
            standardize: bool = False,
            loss_function_critic: tf.keras.losses.Loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM),
            optimizer_common: optimizer_v2.OptimizerV2 = tf.keras.optimizers.Adam(),
            optimizer_actor: optimizer_v2.OptimizerV2 = tf.keras.optimizers.Adam(),
            optimizer_critic: optimizer_v2.OptimizerV2 = tf.keras.optimizers.Adam(),
            n_hidden: int = 64,
            activation: str = 'relu',
    ):

        super().__init__(
            greed_eps, greed_eps_decay, greed_eps_min, alpha_common, alpha_actor, alpha_critic,
            gamma, standardize, loss_function_critic, optimizer_common, optimizer_actor, optimizer_critic,
            n_hidden, activation
        )

        self.env = env
        self.finish_init(0, logger, n_actions)
        self.max_steps_per_episode = max_steps_per_episode

    def train_agent(self, num_episodes):

        for e in range(num_episodes):

            state = self.env.reset()

            self.start_gradient_recording()

            for t in range(self.max_steps_per_episode):

                action = self.act(state, t)
                state, reward, done, _ = self.env.step(action.numpy())

                self.reward(reward, t)

                if tf.cast(done, tf.bool):
                    break

            self.calc_loss_and_update_weights()

            self.logger.print_status(self.env.count_episodes)

    def test_agent(self, num_episodes):

        for e in range(num_episodes):

            state = self.env.reset()

            for t in range(self.max_steps_per_episode):

                action = self.act(state, t)
                state, reward, done, _ = self.env.step(action.numpy())

                self.reward(reward, t)

                if tf.cast(done, tf.bool):
                    break
            self.logger.print_status(self.env.count_episodes)
