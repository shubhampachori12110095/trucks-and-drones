import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from gym import spaces

def transform_box_to_discrete(dims, act_space):
    n = dims^(np.sum(act_space.shape))
    return n, tf.constant(act_space.high, dtype=tf.float32), tf.constant(act_space.low, dtype=tf.float32)

def transform_discrete_to_box(act_space):
    n = tf.cast(tf.constant(act_space.n), dtype=tf.float32)
    act_shape = (1,)
    return act_shape, n

def contin_act_to_discrete(action, act_shape, highs, lows):
    for dim in act_shape:


def discrete_act_to_contin(action, n):
    return tf.cast(tf.cast(action, dtype=tf.float32) * tf.cast(n, dtype=tf.float32), dtype=tf.int32)

class ParallelGradientTape:

    def __init__(
            self,
            layers,
            optimizer,
            grad_alpha=1,
            dynamic_grad_alpha=False,
            dga_rate=0.15
    ):

        self.trainables = [elem.trainable_variables for elem in layers]
        self.optimizer = optimizer
        self.grad_alpha = tf.constant(grad_alpha, dtype=tf.float32)

        self.dynamic_grad_alpha = dynamic_grad_alpha
        if self.dynamic_grad_alpha:
            self.dga_rate = tf.constant(dga_rate, dtype=tf.float32)
            self.one_minus_dga_rate = tf.constant(1 - dga_rate, dtype=tf.float32)

    def reset_tape(self):

        self.tape = tf.GradientTape(persistent=True)
        self.tape._push_tape()
        self.tape.watch(self.trainables)

    def apply_tape(self, loss):

        loss = tf.cast(loss, tf.float32)
        if self.dynamic_grad_alpha:
            self.grad_alpha = self.grad_alpha * self.one_minus_dga_rate + (1 / tf.reduce_max(loss)) * self.dga_rate
        loss = loss*self.grad_alpha

        self.tape._pop_tape()
        grads = self.tape.gradient(loss, [elem.trainable_variables[0] for elem in self.trainables])
        self.optimizer.apply_gradients(zip(grads, [elem.trainable_variables[0] for elem in self.trainables]))
        del self.tape
        return loss


class BaseAgentCore:

    def __init__(
            self,
            gamma: float = 0.9,
            standardize: str = None,
            g_mean: float = 0.0,
            g_rate: float = 0.15,
            discount_rewards: str = None,
            normalize_rewards: str = None,
            reward_range: tuple = None,
            dynamic_rate: float = 0.15,
            auto_transform_actions: bool = True,
    ):

        self.agent_type = 'unspecified'
        self.agent_act_space = 'discrete'

        self.gamma = gamma
        self.eps = tf.constant(np.finfo(np.float32).eps.item(), dtype=tf.float32)

        self.standardize_returns = False
        self.global_standardize_returns = False

        if not standardize is None:

            if standardize =='normal':
                self.standardize_returns = True

            elif standardize == 'global':
                self.global_standardize_returns = True
                self.g_mean = tf.constant(g_mean, dtype=tf.float32)
                self.g_rate = tf.constant(g_rate, dtype=tf.float32)
                self.one_minus_g_rate = tf.constant(1 - g_rate, dtype=tf.float32)

            else:
                raise Exception(
                    "standardize is {}, but must be None, 'normal' or 'global'".format(standardize)
                )

        self.forward_discount_rewards = False
        self.backward_discount_rewards = False

        if not discount_rewards is None:

            if discount_rewards == 'forward':
                self.forward_discount_rewards = True

            elif discount_rewards == 'backward':
                self.backward_discount_rewards = True

            else:
                raise Exception(
                    "discount_rewards is {}, but must be None, 'forward' or 'backward'".format(discount_rewards)
                )

        self.simple_normalize_rewards = False
        self.dynamic_normalize_rewards = False
        self.static_normalize_rewards = False

        if not normalize_rewards is None:

            if normalize_rewards == 'simple':
                self.simple_normalize_rewards = True

            elif normalize_rewards == 'static':
                self.static_normalize_rewards = True

            elif normalize_rewards == 'dynamic':
                self.dynamic_normalize_rewards = True
                self.dynamic_rate = tf.constant(dynamic_rate, dtype=tf.float32)
                self.one_minus_dynamic_rate = tf.constant(1-dynamic_rate, dtype=tf.float32)

            else:
                raise Exception(
                    "normalize_rewards is {}, but must be None, 'simple' 'static' or 'dynamic'".format(normalize_rewards)
                )

        if self.static_normalize_rewards or self.dynamic_normalize_rewards:
            if not reward_range is None:
                self.max_neg_reward = tf.abs(tf.constant(reward_range[0], dtype=tf.float32))
                self.max_pos_reward = tf.abs(tf.constant(reward_range[1], dtype=tf.float32))
            else:
                self.max_neg_reward = tf.abs(tf.constant(-1, dtype=tf.float32))
                self.max_pos_reward = tf.abs(tf.constant(1, dtype=tf.float32))

        self.auto_transform_actions = auto_transform_actions

    def finish_init(self, agent_index, logger, act_space):

        self.agent_index = agent_index
        self.logger = logger

        if self.agent_act_space == 'discrete':

            if isinstance(act_space, spaces.Box):
                if self.auto_transform_actions:
                    input_shape, self.highs, self.lows = transform_box_to_discrete(act_space)
                else:
                    raise Exception(
                        '{} with index {} was assigned to contin actions! Change to spaces.Discrete().'.format(
                            self.agent_type ,self.agent_index
                        )
                    )

            elif isinstance(act_space, spaces.Discrete):
                input_shape = act_space.n

            elif not isinstance(act_space, int):
                input_shape = (act_space,)

            else:
                raise Exception(
                    '{} with index {} was assigned to {} which can not be interpreted! Change to spaces.Box().'.format(
                        self.agent_type ,self.agent_index, act_space))


        elif self.agent_act_space == 'contin':

            if isinstance(act_space, spaces.Box):
                input_shape = act_space.shape

            elif isinstance(act_space, spaces.Discrete):
                if self.auto_transform_actions:
                    input_shape, n = transform_discrete_to_box(act_space)
                else:
                    raise Exception(
                        '{} with index {} was assigned to discrete actions! Change to spaces.Box().'.format(
                            self.agent_type, self.agent_index
                        )
                    )

            elif not isinstance(act_space, int):
                input_shape = (act_space,)

            else:
                raise Exception(
                    '{} with index {} was assigned to {} which can not be interpreted! Change to spaces.Box().'.format(
                        self.agent_type ,self.agent_index, act_space))

        else:
            raise Exception(
                "agent_act_space is {}, but must be 'discrete' or 'contin'".format(self.agent_act_space)
            )

        self.build_nn(input_shape)

    def build_nn(self):

        self.AgentModel =

    def reward(self, rew, t):

        self.rewards = self.rewards.write(t, rew)

    def transform_rewards(self, rewards):

        rewards = tf.cast(rewards, dtype=tf.float32)

        if self.simple_normalize_rewards:
            rewards = rewards/tf.cast(tf.shape(rewards)[0], dtype=tf.float32)

        if self.static_normalize_rewards or self.dynamic_normalize_rewards:
            rewards = tf.where(rewards > 0, rewards / self.max_pos_reward, rewards)
            rewards = tf.where(rewards < 0, rewards / self.max_neg_reward, rewards)

        if self.dynamic_normalize_rewards:

            min_r = tf.reduce_min(rewards)
            if min_r < 0:
                self.max_neg_reward = self.dynamic_rate * tf.abs(
                    min_r) + self.one_minus_dynamic_rate * self.max_neg_reward

            max_r = tf.reduce_max(rewards)
            if max_r > 0:
                self.max_pos_reward = self.dynamic_rate * max_r + self.one_minus_dynamic_rate * self.max_pos_reward

        if self.forward_discount_rewards:

            sample_len = tf.shape(rewards)[0]
            placeholder_rewards = tf.TensorArray(dtype=tf.float32, size=sample_len)

            for i in tf.range(sample_len):
                placeholder_rewards = placeholder_rewards.write(rewards[i] / tf.cast(i, dtype=tf.float32))

            rewards = placeholder_rewards.stack()

        if self.backward_discount_rewards:

            sample_len = tf.shape(rewards)[0]
            placeholder_rewards = tf.TensorArray(dtype=tf.float32, size=sample_len)
            rewards = rewards[::-1]

            for i in tf.range(sample_len):
                placeholder_rewards = placeholder_rewards.write(rewards[i] / tf.cast(i, dtype=tf.float32))

            rewards = placeholder_rewards.stack()[::-1]

        return rewards


    def get_expected_return(self, rewards):

        sample_len = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=sample_len)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)

        discounted_sum = tf.constant(0.0, dtype=tf.float32)
        discounted_sum_shape = discounted_sum.shape

        for i in tf.range(sample_len):

            discounted_sum = rewards[i] + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1]

        if self.standardize_returns:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + self.eps))

        if self.global_standardize_returns:
            self.global_mean = tf.math.reduce_mean(returns) * self.g_rate + self.global_mean * self.one_minus_g_rate
            returns = ((returns - (self.global_mean / 2)) /
                    (tf.math.reduce_std(returns) + self.eps))

        return returns

