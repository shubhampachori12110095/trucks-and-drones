"""Dummy Agent where actions are always an empty list."""


class DummyAgent:

    def __init__(self, env, name='DummyAgent', max_steps_per_episode=1000):

        self.name = name
        self.env = env
        self.max_steps_per_episode = max_steps_per_episode

    def train_agent(self, num_episodes):

        for e in range(num_episodes):
            self.env.reset()

            for i in range(self.max_steps_per_episode):
                _, _, done, _ = self.env.step([])

                if done:
                    break

    def test_agent(self, num_episodes):

        self.train_agent(num_episodes)
