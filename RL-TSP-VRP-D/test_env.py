

from main.build_env import BuildEnvironment
from main.agents.core_agent import DummyAgent

env = BuildEnvironment('test', debug_mode=False)

env.trucks(1)
env.drones(1)
env.robots(1)

env.depots(1)
env.customers([10, 20])

env.dummy_observations()
env.dummy_actions()

env.compile()

agent = DummyAgent(env.build())

agent.test_agent(1000)
