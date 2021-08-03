import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
import numpy as np


class MultiAgent:

    def __init__(
            self,
            name,
            env,
            agents,
            common_model,
            logger,
            agents_with_target_models: list,
            agents_with_q_future: list,
            optimizer: optimizer_v2.OptimizerV2,
            tau: float,
            use_target_model: bool = True,
            target_update: int = 1,
            max_steps_per_episode: int = 1000,
            actions_as_list = False,
            done_penalty = 0,
            sum_rewards = False,
    ):

        self.name = name
        self.env = env
        self.agents = agents
        self.model = common_model
        self.logger = logger
        self.optimizer = optimizer
        self.tau = tau
        self.use_target_model = use_target_model
        self.target_update = target_update

        self.save_interval = 100
        self.save_path = '/models'

        if self.use_target_model:
            self.target_model = tf.keras.models.clone_model(self.model)

        self.max_steps_per_episode = max_steps_per_episode

        self.actions_as_list = actions_as_list
        self.done_penalty = done_penalty
        self.sum_rewards = sum_rewards

        self.agents_with_target_models = agents_with_target_models
        self.agents_with_q_future = agents_with_q_future

    #  @tf.function
    def train_agent(self, num_episodes):

        for e in range(num_episodes):

            state = self.env.reset()
            info_dict = None
            sum_r = 0

            with tf.GradientTape() as tape:

                tape.watch(self.model.trainable_variables)

                [agent.start_gradient_recording() for agent in self.agents]

                done = False
                t = 0
                while not done:

                    common_outputs = self.model_pred(state)
                    actions_list = [agent.act(common_outputs, t, info_dict) for agent in self.agents]
                    if self.actions_as_list:
                        state, reward, done, info_dict = self.env.step([elem.numpy() for elem in actions_list])
                    else:
                        state, reward, done, info_dict = self.env.step(
                            np.squeeze(np.array([elem.numpy() for elem in actions_list]))
                        )
                    self.env.render()

                    if self.sum_rewards:
                        sum_r += reward
                        reward = sum_r

                    if not self.max_steps_per_episode is None:
                        if t > self.max_steps_per_episode:
                            done = True

                    if done:
                        self.assign_rewards(t, reward - self.done_penalty)
                    else:
                        self.assign_rewards(t, reward)

                    if len(self.agents_with_q_future) > 0:

                        if tf.cast(self.use_target_model, tf.bool):
                            target_outputs = self.target_model_pred(state)

                        else:
                            target_outputs = self.model_pred(state)

                        [self.agents[i].pred_q_future(target_outputs, t, done) for i in self.agents_with_q_future]

                    t += 1
                    if tf.cast(done, tf.bool):
                        break

                loss = sum([agent.calc_loss_and_update_weights() for agent in self.agents])

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if e % self.target_update == 0:
                self.update_target_weights()

            try:
                self.logger.print_status(self.env.count_episodes)
            except:
                self.logger.print_status(e)


            #if e == num_episodes - 1 or e % self.save_interval == 0:
                #self.model.save(self.save_path)

    def assign_rewards(self, t, reward):
        if isinstance(reward, list):
            [self.agents[i].reward(t, reward[i]) for i in range(len(self.agents))]
        else:
            [agent.reward(t, reward) for agent in self.agents]

    def model_pred(self, inputs):
        if isinstance(inputs, list):
            return self.model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in inputs])
        return self.model(tf.expand_dims(tf.convert_to_tensor(inputs), 0))

    def target_model_pred(self, inputs):
        if isinstance(inputs, list):
            return self.target_model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in inputs])
        return self.target_model(tf.expand_dims(tf.convert_to_tensor(inputs), 0))

    def update_target_weights(self):

        if self.use_target_model:
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
            self.target_model.set_weights(target_weights)

        [self.agents[i].update_target_weights() for i in self.agents_with_target_models]


    def env_step(self, action: np.ndarray):
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action):
        return tf.numpy_function(self.env_step, [action],
                                 [tf.float32, tf.int32, tf.int32])



