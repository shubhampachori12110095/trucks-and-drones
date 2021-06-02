import numpy as np
import gym
import pandas as pd
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from datetime import datetime
from collections import deque


from env import CustomEnv



class DQN:
    def __init__(self, env=CustomEnv()):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                # sum(actor.action_prob*self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma # + GLOBAL REWARD falls für diese stelle relevant
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env = CustomEnv()

    epoch_len = 1000
    epochs    = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for e in range(epochs):
        cur_state = env.reset()#.reshape(1,2)
        print(np.shape(cur_state))
        
        for step in range(epoch_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            #new_state = new_state#.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            if step % 100 == 0:
                dqn_agent.replay()       # internally iterates default (prediction) model
                dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                break

        if e % 10 == 0:
            dqn_agent.save_model("/models/epoch-{}.model".format(e))


if __name__ == "__main__":
    main()




