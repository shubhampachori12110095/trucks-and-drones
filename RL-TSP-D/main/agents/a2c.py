import numpy as np
import tensorflow as tf

import keras
from keras import layers

from collections import deque


class ActorCrticNetwork:

    def __init__(self,):

        self.discrete_activation = 'softmax'
        self.hidden_activation   = 'relu'
        self.list_input_shapes   = [[5,6], [3]]
        self.num_critic_outputs  = 1
        self.list_actor_outputs  = [30,2]



        num_inputs = [[5,6],[3]]
        num_actions = 60
        num_hidden = 30

    def create_model(self):

        #3 input layer? 
        input_1 = layers.Input(shape=(30,6,))
        input_2 = layers.Input(shape=(3,))
        #input_3 = Input(shape=(2,))

        flatten_0     = layers.Flatten()(input_1)
        dense_layer_0 = layers.Dense(30, activation='relu')(flatten_0)
        combined = layers.Concatenate(axis=-1)([dense_layer_0,input_2])#,input_3])
        common = layers.Dense(num_hidden, activation="relu")(combined)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        model = keras.Model(inputs=[input_1, input_2], outputs=[action, critic])


def a2c_parameter(
        max_steps_episode = 1000,
        gamma             = 0.99,
        lr                = 0.01,
        epsilon           = 1.0,
        epsilon_dec       = 0.999,
        epsilon_min       = 0.01,
        run_factor        = 0.05,
        max_run_reward    = 1,
        optimizer         = 'adam',
        critic_loss       = 'huber',
        ):
    return {
        'max_steps_episode': max_steps_episode,
        'gamma':             gamma,
        'lr':                lr,
        'epsilon':           epsilon,
        'epsilon_dec':       epsilon_dec,
        'epsilon_min':       epsilon_min,
        'run_factor':        run_factor,
        'max_run_reward':    max_run_reward,
        'optimizer':         optimizer,
        'critic_loss':       critic_loss
        }



class MultiA2C:

    def __init__(self, env, model, a2c_param=a2c_parameter()):

        [setattr(self, k, param_interpret(v)) for k, v in a2c_param.items()]

        self.env    = env
        self.model  = model
        self.memory = deque()

        if self.optimizer == 'adam':
            self.otimizer = keras.optimizers.Adam(self.lr)

        if self.critic_loss == 'huber':
            self.critic_loss = keras.loss.Huber()

        self.num_actions = 

        self.smallest_val = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    

    def greedy_epsilon(self, action):
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions, p=np.squeeze(action_probs))

        return np.argmax(np.squeeze(action_probs))


    def expected_rewards(self, rewards_hist):
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_hist[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.smallest_val)
        return returns.tolist()


    def calc_loss(self,action_prob_hist,state_value_hist,rewards_hist):

        # Calculating loss values to update our network
        actor_losses = []
        critic_losses = []
        
        for log_prob, value, exp_r in zip(action_prob_hist, state_value_hist, expected_rewards(rewards_hist)):
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = exp_r - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(exp_r, 0)))

        return sum(actor_losses) + sum(critic_losses)
        


    def train_agent(self):

        action_prob_hist = []
        state_value_hist = []
        rewards_hist     = []
        
        running_reward = 0

        while True:

            epoch_reward = 0
            state = env.reset()
            with tf.GradientTape() as grad_tape:

                for step in range(max_steps_per_episode):

                    action_prob, state_value = self.model(state)

                    action = greedy_epsilon(action_prob)                    

                    state, reward, done, _ = env.step(action)

                    action_prob_hist.append(tf.math.log(action_prob[0, action]))
                    state_value_hist.append(state_value[0, 0])
                    rewards_hist.append(reward)

                    episode_reward += reward

                    if done:
                        break

                # Update running reward to check condition for solving
                running_reward = self.run_factor * episode_reward + (1 - self.run_factor) * running_reward

                sum_loss = self.calc_loss(action_prob_hist,state_value_hist,rewards_hist)

                # Backpropagation
                grads = grad_tape.gradient(sum_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Clear the loss and reward history
                action_prob_hist.clear()
                state_value_hist.clear()
                rewards_hist.clear()

                # Log details
                if self.env.count_epochs % 10 == 0:
                    template = "running reward: {:.2f} at episode {}"
                    print(template.format(running_reward, self.env.count_epochs))

                if running_reward > self.max_run_reward:  # Condition to consider the task solved
                    print("Solved at episode {}!".format(self.env.count_epochs))
                    break


    def test_agent(self, num_epochs=10):

        for epoch in range(num_epochs):
            
            state = env.reset()
            done = False
            
            while not done:
                action_prob, state_value = self.model(state)
                action = np.argmax(np.squeeze(action_probs))
                state, reward, done, _ = env.step(action)

            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, self.env.count_epochs))



class BaseA2C:

    def __init__(self, env, model, a2c_param=a2c_parameter()):

        [setattr(self, k, param_interpret(v)) for k, v in a2c_param.items()]

        self.env    = env
        self.model  = model
        self.memory = deque()

        if self.optimizer == 'adam':
            self.otimizer = keras.optimizers.Adam(self.lr)

        if self.critic_loss == 'huber':
            self.critic_loss = keras.loss.Huber()

        self.num_actions = 

        self.smallest_val = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    

    def greedy_epsilon(self, action):
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions, p=np.squeeze(action_probs))

        return np.argmax(np.squeeze(action_probs))


    def expected_rewards(self, rewards_hist):
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_hist[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.smallest_val)
        return returns.tolist()


    def calc_loss(self,action_prob_hist,state_value_hist,rewards_hist):

        # Calculating loss values to update our network
        actor_losses = []
        critic_losses = []
        
        for log_prob, value, exp_r in zip(action_prob_hist, state_value_hist, expected_rewards(rewards_hist)):
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = exp_r - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(exp_r, 0)))

        return sum(actor_losses) + sum(critic_losses)
        


    def train_agent(self):

        action_prob_hist = []
        state_value_hist = []
        rewards_hist     = []
        
        running_reward = 0

        while True:

            epoch_reward = 0
            state = env.reset()
            with tf.GradientTape() as grad_tape:

                for step in range(max_steps_per_episode):

                    action_prob, state_value = self.model(state)

                    action = greedy_epsilon(action_prob)                    

                    state, reward, done, _ = env.step(action)

                    action_prob_hist.append(tf.math.log(action_prob[0, action]))
                    state_value_hist.append(state_value[0, 0])
                    rewards_hist.append(reward)

                    episode_reward += reward

                    if done:
                        break

                # Update running reward to check condition for solving
                running_reward = self.run_factor * episode_reward + (1 - self.run_factor) * running_reward

                sum_loss = self.calc_loss(action_prob_hist,state_value_hist,rewards_hist)

                # Backpropagation
                grads = grad_tape.gradient(sum_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Clear the loss and reward history
                action_prob_hist.clear()
                state_value_hist.clear()
                rewards_hist.clear()

                # Log details
                if self.env.count_epochs % 10 == 0:
                    template = "running reward: {:.2f} at episode {}"
                    print(template.format(running_reward, self.env.count_epochs))

                if running_reward > self.max_run_reward:  # Condition to consider the task solved
                    print("Solved at episode {}!".format(self.env.count_epochs))
                    break


    def test_agent(self, num_epochs=10):

        for epoch in range(num_epochs):
            
            state = env.reset()
            done = False
            
            while not done:
                action_prob, state_value = self.model(state)
                action = np.argmax(np.squeeze(action_probs))
                state, reward, done, _ = env.step(action)

            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, self.env.count_epochs))


