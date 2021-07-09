

from main.build_env import BuildEnvironment
from main.agents.agents import DummyAgent
from main.agents.build_agent import BaseAgentBuilder

env = BuildEnvironment('test', debug_mode=False)

env.trucks(3)
env.drones(3, max_cargo=2)
env.robots(1)

env.depots(1)
env.customers([10, 20])

env.observations()
env.actions(
    mode = 'single_vehicle',  # 'multi_vehicle'
    flattened= 'per_output',  # 'per_vehicle', #'all'
    contin_outputs= [],
    discrete_outputs = ['nodes', 'coord'],
    binary_discrete= ['move', 'v_load_sep_unload'],
    binary_contin= [],
    num_discrete_bins = 20,
    combine = 'contin',  # 'discrete', 'by_categ', 'all', list of lists of output names
)

env.compile()

#agent = DummyAgent(env.build())
agent = BaseAgentBuilder(env.build(), 'test')

print(agent.action_outputs)
[agent.assign_agent_to_act(act_index=i) for i in range(len(agent.action_outputs))]
print(agent.action_outputs)

agent.compile_agents()
agent.seperate_input_layer()
agent.add_hidden_to_each_input()
agent.add_combined_layer()
agent.add_sum_output_neurons_layer()
agent.add_output_networks()

agent = agent.build()

agent.train_agent(1000)
