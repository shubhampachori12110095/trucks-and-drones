import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
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


class MultiAgent:

    def __init__(
        self,
        name,
        env,
        agents,
        model,
        logger,
        out_ind: list,
        q_out_ind: list,
        optimizer: optimizer_v2.OptimizerV2,
        tau: float,
        use_target_model: bool = True,
        target_update: int = 1,
        max_steps_per_episode: int = 1000
    ):

        self.name = name
        self.env = env
        self.agents = agents
        self.model = model
        self.logger = logger
        self.optimizer = optimizer
        self.tau = tau
        self.use_target_model = use_target_model
        self.target_update = target_update

        self.save_interval = 100
        self.save_path = '/models'

        if self.use_target_model:
            self.target_model = tf.keras.models.clone_model(model)

        self.max_steps_per_episode = max_steps_per_episode

        self.out_ind = out_ind
        self.q_ind = [agent.agent_index for agent in self.agents if agent.agent_type == 'dqn' or agent.agent_type == 'sac']
        self.q_out_ind = q_out_ind

    #@tf.function
    def train_agent(self, num_episodes):

        for e in range(num_episodes):

            state = self.env.reset()

            with tf.GradientTape() as tape:

                for t in range(self.max_steps_per_episode):

                    outputs_list = self.model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])
                    #print('outputs_list')
                    #print(outputs_list)
                    actions_list = [self.agents[i].act([outputs_list[j] for j in self.out_ind[i]], t) for i in range(len(self.agents))]
                    actions_list = [elem.numpy() for elem in actions_list]
                    #print(actions_list)
                    state, rewards, done, _ = self.env.step(actions_list)
                    self.env.render()
                    [self.agents[i].reward(rewards-t*10, t) for i in range(len(self.agents))]
                    #print('reward',rewards)

                    if tf.cast(self.use_target_model, tf.bool):
                        if tf.cast(done, tf.bool):
                            [self.agents[i].q_future(0, t) for i in self.q_ind]
                        else:
                            outputs_list = [self.target_model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])]
                            #print('target outputs')
                            #print(outputs_list)
                            q_future_list = [tf.math.reduce_max(tf.squeeze(elem)) for elem in outputs_list]
                            [self.agents[i].q_future([q_future_list[j] for j in self.q_out_ind[i]], t) for i in self.q_ind]

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

            self.logger.print_status(self.env.count_episodes)

            #if e == num_episodes - 1 or e % self.save_interval == 0:
                #self.model.save(self.save_path)


    def env_step(self, action: np.ndarray):
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action):
        return tf.numpy_function(self.env_step, [action],
                                 [tf.float32, tf.int32, tf.int32])



