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

    def __init__(self, name, env, agents, model, out_ind, q_out_ind, optimizer, tau, use_target_model=True, target_update=1, max_steps_per_episode=1000):

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
                    actions_list = [self.agents[i].act([outputs_list[j] for j in self.out_ind[i]], t) for i in range(len(self.agents))]
                    actions_list = [elem.numpy() for elem in actions_list]
                    print(actions_list)
                    state, rewards, done, _ = self.env.step(actions_list)
                    self.env.render()
                    [self.agents[i].reward(rewards, t) for i in range(len(self.agents))]

                    if tf.cast(self.use_target_model, tf.bool):
                        if tf.cast(done, tf.bool):
                            [self.agents[i].q_future(0, t) for i in self.q_ind]
                        else:
                            outputs_list = [self.target_model([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in state])]
                            q_future_list = [tf.math.reduce_max(tf.squeeze(elem)) for elem in actions_list]
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



class DQNAgent:

    def __init__(self, params):

        self.agent_type = 'dqn'
        self.agent_index = params['act_index']
        self.activation_q = params['activation_q']
        self.loss_q = params['loss_q']
        self.greed_eps = params['greed_eps']
        self.greed_eps_decay = params['greed_eps_decay']
        self.greed_eps_min = params['greed_eps_min']
        self.gamma_q = params['gamma_q']
        self.alpha = params['alpha_q']

        self.alpha = 1
        self.gamma_q = 0.99

        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.targets = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.Q_futures = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)

        self.num_actions = 0

    def indices(self):
        return [self.agent_index + i for i in range(3)]

    def act(self, actions, t):
        self.greed_eps *= self.greed_eps_decay
        self.greed_eps = max(self.greed_eps, self.greed_eps_min)
        actions = tf.squeeze(actions[0])
        #print(actions)
        #actions = tf.make_ndarray(actions)

        if np.random.random() < self.greed_eps:
            action = tf.convert_to_tensor(np.random.choice(self.num_actions))
        else:
            action = tf.math.argmax(actions)

        print('action',action)
        print(actions)
        print(actions[action])

        self.targets = self.targets.write(t, actions[action])

        return action

    def reward(self, rew, t):
        self.rewards = self.rewards.write(t, rew)

    def q_future(self, q_fut, t):
        #print(q_fut)
        self.Q_futures = self.Q_futures.write(t, tf.cast(tf.squeeze(q_fut), dtype=tf.float32))

    def calc_loss(self):
        self.targets = self.targets.stack()
        self.rewards = self.rewards.stack()
        self.Q_futures = self.Q_futures.stack()

        loss =  self.alpha * self.loss_function(self.targets, (self.rewards + tf.squeeze(self.Q_futures) * self.gamma_q))

        self.targets = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.Q_futures = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        return loss


class DiscreteA2CAgent:

    def __init__(self, params):

        self.agent_type = 'dqn'
        self.agent_index = params['act_index']
        self.activation_grad = params['activation_grad']
        self.activation_val = params['activation_val']
        self.loss_grad = params['loss_grad']
        self.loss_val = params['loss_val']
        self.greed_eps = params['greed_eps']
        self.greed_eps_decay = params['greed_eps_decay']
        self.greed_eps_min = params['greed_eps_min']
        self.gamma_v= params['gamma_v']
        self.alpha_grad = params['alpha_grad']
        self.alpha_val = params['alpha_val']


        self.loss_val = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.values = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True)
        self.act_probs = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True)

    def act(self, actions, t):

        action_logits_t, value = actions[0], actions[1]

        # Sample next action from the action probability distribution

        action_probs_t = tf.nn.softmax(action_logits_t)

        self.greed_eps *= self.greed_eps_decay
        self.greed_eps = max(self.greed_eps, self.greed_eps_min)

        if np.random.random() < self.greed_eps:
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
        else:
            action = tf.math.argmax(action_logits_t)

        self.values = self.values.write(t, tf.squeeze(value))
        self.act_probs = self.act_probs.write(t,tf.nn.softmax(action_logits_t)[0, action])
        return action

    def reward(self, rew):
        self.rewards = self.rewards.write(t, rew)

    def get_expected_return(self):
        """Compute expected returns per timestep."""

        n = tf.shape(self.rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(self.rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if self.standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + eps))

        return returns

    def calc_loss(self):
        self.values = self.targets.stack()
        self.rewards = self.rewards.stack()
        self.act_probs = self.Q_futures.stack()

        # Calculate expected returns
        returns = self.get_expected_return()

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [self.act_probs, self.values, returns]]

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = huber_loss(values, returns)

        return self.alpha_val*critic_loss + self.alpha_grad*actor_loss












