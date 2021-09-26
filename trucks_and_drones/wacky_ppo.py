import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, LSTM, Dropout

from wacky_rl.agents import AgentCore
from wacky_rl.models import WackyModel
from wacky_rl.memory import BufferMemory

from wacky_rl.losses import PPOActorLoss, MeanSquaredErrorLoss
from wacky_rl.transform import GAE
from wacky_rl.trainer import Trainer

from wacky_rl.transform import RunningMeanStd

from wacky_rl.logger import StatusPrinter


class WackyPPO(AgentCore):


    def __init__(
            self,
            env,
            epochs=10,
            batch_size=64,
            learning_rate=3e-4,
            clipnorm=0.5,
            entropy_factor=0.0,
            hidden_units=64,
            hidden_activation='relu',
            kernel_initializer='glorot_uniform',
            logger=None,
            approximate_contin=False,
    ):
        super(WackyPPO, self).__init__()

        self.epochs = epochs
        self.batch_size = batch_size

        self.logger = logger

        self.approximate_contin = approximate_contin
        self.memory = BufferMemory()
        self.advantage_and_returns = GAE()

        self.reward_rmstd = RunningMeanStd()

        kernel_initializer = tf.keras.initializers.Orthogonal()

        #print(env.observation_space.shape)
        #input_layer = Input((3,34))
        input_layer = Input(env.observation_space.shape)
        #lstm_layer = LSTM(32, kernel_initializer=kernel_initializer)(input_layer)
        hidden_layer = Dense(hidden_units, activation=hidden_activation, kernel_initializer=kernel_initializer)(input_layer)
        hidden_layer = Dropout(0.1)(hidden_layer)
        hidden_layer = Dense(hidden_units, activation=hidden_activation, kernel_initializer=kernel_initializer)(hidden_layer)
        hidden_layer = Dropout(0.1)(hidden_layer)

        action_layer = Dense(64, activation=hidden_activation, kernel_initializer=kernel_initializer)(hidden_layer)
        action_layer = Dense(32, activation=hidden_activation, kernel_initializer=kernel_initializer)(action_layer)
        action_layer = self.make_action_layer(env, approx_contin=approximate_contin, kernel_initializer=kernel_initializer)(action_layer)
        critic_layer = Dense(64, activation=hidden_activation, kernel_initializer=kernel_initializer)(hidden_layer)
        critic_layer = Dense(32, activation=hidden_activation, kernel_initializer=kernel_initializer)(critic_layer)
        critic_layer = Dense(1, kernel_initializer=kernel_initializer)(critic_layer)

        self.model = WackyModel(inputs=input_layer, outputs=[action_layer, critic_layer])

        self.actor_loss = PPOActorLoss(entropy_factor=entropy_factor)
        self.critic_loss = MeanSquaredErrorLoss()
        if clipnorm is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=clipnorm)

    def act(self, inputs, act_argmax=False, save_memories=True):

        dist, val = self.model(tf.expand_dims(tf.squeeze(inputs), 0))

        if act_argmax:
            actions = dist.mean_actions()
        else:
            actions = dist.sample_actions()

        probs = dist.calc_probs(actions)

        if act_argmax:
            self.logger.log_mean('argmax probs', probs)
        else:
            self.logger.log_mean('probs', probs)

        if save_memories:
            self.memory(actions, key='actions')
            self.memory(probs, key='probs')
            self.memory(val, key='val')

        return self.transform_actions(dist, actions)

    def learn(self):

        '''action, old_probs, values, states, new_states, rewards, dones = self.memory.replay()

        #self.reward_rmstd.update(rewards.numpy())
        #rewards = rewards / np.sqrt(self.reward_rmstd.var + 1e-8)

        _ , next_value = self.model(tf.expand_dims(new_states[-1], 0))
        adv, ret = self.advantage_and_returns(rewards, dones, values, next_value)

        self.logger.log_mean('values', np.mean(np.append(values, next_value)))
        self.logger.log_mean('adv', np.mean(adv.numpy()))
        self.logger.log_mean('ret', np.mean(ret.numpy()))


        for i in range(len(adv)):
            self.memory(adv[i], key='adv')
            self.memory(ret[i], key='ret')'''

        a_loss_list = []
        c_loss_list = []
        sum_loss_list = []

        for e in range(self.epochs):

            action, old_probs, values, states, new_states, rewards, dones = self.memory.replay()

            _, next_value = self.model(tf.expand_dims(new_states[-1], 0))
            adv, ret = self.advantage_and_returns(rewards, dones, values, next_value)

            self.logger.log_mean('values', np.mean(np.append(values, next_value)))
            self.logger.log_mean('adv', np.mean(adv.numpy()))
            self.logger.log_mean('ret', np.mean(ret.numpy()))

            for i in range(len(adv)):
                self.memory(adv[i], key='adv')
                self.memory(ret[i], key='ret')

            with tf.GradientTape() as tape:

                grad_loss_list = []
                for mini_batch in self.memory.mini_batches(batch_size=64, num_batches=None, shuffle_batches=False):

                    actions, old_probs, values, states, new_states, rewards, dones, adv, ret = mini_batch

                    adv = tf.squeeze(adv)
                    adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

                    #with tf.GradientTape() as tape:

                    pred_dist, pred_val = self.model(states)

                    a_loss = self.actor_loss(pred_dist, actions, old_probs, adv)
                    c_loss = self.critic_loss(pred_val, ret)

                    a_loss_list.append(np.mean(a_loss.numpy()))
                    c_loss_list.append(np.mean(c_loss.numpy()))

                    sum_loss = a_loss + 0.5 * c_loss

                    sum_loss_list.append(np.mean(sum_loss.numpy()))
                    grad_loss_list.append(sum_loss)

                sum_loss = tf.reduce_mean(tf.stack(grad_loss_list))

            if not tf.math.is_nan(sum_loss):
                grad = tape.gradient(sum_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

                #self.optimizer.minimize(sum_loss, self.model.trainable_variables, tape=tape)

            self.memory.pop_array('adv')
            self.memory.pop_array('ret')


        self.logger.log_mean('actor_loss', np.mean(a_loss_list))
        self.logger.log_mean('critic_loss', np.mean(c_loss_list))
        self.logger.log_mean('sum_loss', np.mean(sum_loss_list))

        self.memory.clear()

    def save_model(self, path='test'):
        self.model.save_weights(path)
        #self.model.save(path)

    def load_model(self, path='test'):
        self.model.load_weights(path)


def train_ppo():

    import gym
    #env = gym.make('CartPole-v0')
    env = gym.make("LunarLanderContinuous-v2")

    agent = WackyPPO(env, logger=StatusPrinter('WackyPPO'))

    trainer = Trainer(env, agent)
    trainer.n_step_train(5_000, train_on_test=False)
    trainer.agent.save_model()
    trainer.test(100)

if __name__ == "__main__":
    train_ppo()