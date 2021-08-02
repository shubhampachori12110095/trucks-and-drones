import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from gym import spaces

from main.agents.core_agent import BaseAgentCore, ParallelGradientTape


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


class DiscreteActorCriticCore(BaseAgentCore):

    def __init__(
            self,
            model_layer,
            gamma: float = 0.9,
            standardize: str = None,
            g_mean: float = 0.0,
            g_rate: float = 0.15,
            discount_rewards: str = None,
            normalize_rewards: str = None,
            reward_range: tuple = None,
            dynamic_rate: float = 0.15,
            auto_transform_actions: bool = True,

            alpha_common: float = 1.0,
            alpha_actor: float = 0.5,
            alpha_critic: float = 0.5,

            loss_function_critic: tf.keras.losses.Loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM),

            tape_common: ParallelGradientTape = None,
            tape_actor: ParallelGradientTape = None,
            tape_critic: ParallelGradientTape = None,

    ):

        super().__init__(
            model_layer, gamma, standardize, g_mean, g_rate, discount_rewards, normalize_rewards,
            reward_range, dynamic_rate, auto_transform_actions
        )

        self.agent_type = 'discrete_a2c'
        self.agent_act_space = 'discrete'
        self.uses_q_future = False
        self.use_target_model = False

        self.alpha_common = alpha_common
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        self.loss_function_critic = loss_function_critic

        self.tape_common = tape_common
        self.tape_actor = tape_actor
        self.tape_critic = tape_critic

    def loss_function_actor(self, act_probs, advantage, loss_critic):

        act_log_probs = tf.math.log(act_probs)
        loss_actor = -tf.math.reduce_sum(act_log_probs * advantage)
        return loss_actor

    def start_gradient_recording(self):

        self.tape_common.reset_tape(self.model.common)
        self.tape_critic.reset_tape(self.model.critic)
        self.tape_actor.reset_tape(self.model.actor)

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


    def calc_loss_and_update_weights(self):

        vals = tf.squeeze(self.values.stack())


        act_probs = tf.squeeze(self.act_probs.stack())
        self.logger.log_mean('act_probs_' + str(self.agent_index), np.mean(act_probs.numpy()))

        rewards = self.transform_rewards()
        ret = self.get_expected_return(rewards)

        loss_critic = self.loss_function_critic(tf.expand_dims(vals, 1), tf.expand_dims(ret, 1))
        loss_critic = self.tape_critic.apply(loss_critic, self.model.critic)


        loss_actor = self.loss_function_actor(tf.expand_dims(act_probs, 1), tf.expand_dims(ret - vals, 1), loss_critic)
        loss_actor = self.tape_actor.apply(loss_actor, self.model.actor)
        self.logger.log_mean('loss_actor_' + str(self.agent_index), loss_actor.numpy())

        loss_common = (self.alpha_actor * loss_actor) + (self.alpha_critic * loss_critic)
        loss_common = self.tape_actor.apply(loss_common, self.model.common)
        self.logger.log_mean('loss_common_' + str(self.agent_index), loss_common.numpy())

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
