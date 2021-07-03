'''

'''
class DummyAgent:

    def __init__(self, env, name='DummyAgent'):

        self.name = name
        self.env = env

    def train_agent(self, num_episodes):

        actions = []

        for e in range(num_episodes):

            state = self.env.reset()
            done = False

            while not done:
                state, reward, done, _ = self.env.step(actions)

    def test_agent(self, num_episodes):

        actions = []

        for e in range(num_episodes):

            state = self.env.reset()
            done = False

            while not done:
                self.env.render()
                state, reward, done, _ = self.env.step(actions)

'''
class BaseAgent:

    def __init__(self, env)
        
'''