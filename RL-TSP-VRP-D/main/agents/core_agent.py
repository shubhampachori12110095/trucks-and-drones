

class Agent:

    def __init__(self, env, name='DummyActor'):

        self.name = name
        self.env = env

    def train_agent(self, num_episodes):

        actions = []

        for e in num_episodes:

            state = env.reset()
            done = False

            while not done:
                state, reward, done, _ = env.step(actions)



    def test_agent(self, num_episodes):

        actions = []

        for e in num_episodes:

            state = env.reset()
            done = False

            while not done:
                state, reward, done, _ = env.step(actions)


