

from main.build_env import BuildEnvironment
from main.agents._old_agents import DummyAgent
from main.agents._old_build_agent import BaseAgentBuilder
from main.logger import TrainingLogger




env = BuildEnvironment('test', debug_mode=False)

env.trucks(1)
#env.drones(3, max_cargo=2)
#env.robots(1)

env.depots(1)
env.customers(10)

env.observations()
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
agent = BaseAgentBuilder(env.build(), log_dir='_logs')

[agent.assign_agent_to_act(act_index=i) for i in range(len(agent.action_outputs))]

agent.compile_agents()
agent.seperate_input_layer()
agent.add_hidden_to_each_input()
agent.add_combined_layer()
agent.add_sum_output_neurons_layer()
agent.add_output_networks()

agent = agent.build(max_steps_per_episode=1000)

agent.train_agent(60000)
