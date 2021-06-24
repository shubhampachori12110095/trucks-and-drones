

from main.build_env import BuildEnvironment
from main.agents.core_agent import DummyAgent

env = BuildEnvironment('test')

env.trucks(4)
env.drones(2)
env.robots(2)

env.depots(2)
env.customers([10,20])

env.dummy_observations()
env.dummy_actions()

env.compile()

agent = DummyAgent(env.build())

agent.test_agent(1000)
