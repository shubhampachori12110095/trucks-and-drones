from trucks_and_drones.build_env import BuildEnvironment
#  from main.agents.dummy_agent import DummyAgent
from old.agents.build_agent import BaseAgentBuilder
from old.agents.discrete_actor_critic import DiscreteActorCriticCore

env = BuildEnvironment('test', debug_mode=False)

env.trucks(1)
#env.drones(3, max_cargo=2)
#env.robots(1)

env.depots(1)
env.customers(10)

env.observations(
    contin_inputs = [['v_coord'], ['c_coord'], ['d_coord'], ['demand']],
    discrete_inputs = None,
)

env.actions(
    mode = 'single_vehicle',  # 'multi_vehicle'
    flattened= 'per_output',  # 'per_vehicle', #'all'
    contin_outputs= [],
    discrete_outputs = ['nodes'],
    binary_discrete= [],
    binary_contin= [],
    num_discrete_bins = 20,
    combine = 'contin',  # 'discrete', 'by_categ', 'all', list of lists of output names
)

env.compile()

#agent = DummyAgent(env.build())
agent = BaseAgentBuilder(env.build(), log_dir='trucks_and_drones/_logs')

[agent.assign_agent(DiscreteActorCriticCore(),at_index=i) for i in range(len(agent.action_outputs))]

agent = agent.build(max_steps_per_episode=200)

agent.train_agent(1000000)
