
from main.simulation.simulation import simulation_parameter
from main.simulation.vehicles import battery_parameter, range_parameter, travel_parameter, cargo_parameter
from main.simulation.nodes import customer_parameter, depot_parameter
from main.simulation.action_interpreter import output_parameter
from main.simulation.state_interpreter import input_parameter
from main.reward_calculator import reward_parameter
from main.visualizer import visual_parameter

from main.environment import CustomEnv
from main.agents.core_agent import Agent

# Vehicles Parameter:
# ----------------------------------------------------------------------------------------------------------------

standard_battery = battery_parameter(
    max_charge         = 100,
    charge_per_step    = 50,
    discharge_per_step = 10,
    init_value         = 0,
    signal_list        = [1,1,-1])


standard_range = range_parameter(
    max_range   = None,
    init_value  = 0,
    signal_list = [1,1,-1])


street_travel = travel_parameter(
    travel_type = 'street',
    speed       = 1)


arial_travel = travel_parameter(
    travel_type = 'arial',
    speed       = 1)


standard_MV_cargo = cargo_parameter(
    cargo_type         = 'standard+extra',
    max_cargo          = 10,
    max_cargo_UV       = 1,
    cargo_per_step     = 10,
    cargo_UV_per_step  = 1,
    init_value         = 0,
    signal_list        = [1,1,-1])


standard_UV_cargo = cargo_parameter(
    cargo_type         = 'standard',
    max_cargo          = 1,
    max_cargo_UV       = 0,
    cargo_per_step     = 1,
    cargo_UV_per_step  = 0,
    init_value         = 0,
    signal_list        = [1,1,-1])


# Nodes Prameter:
# ----------------------------------------------------------------------------------------------------------------

standard_customers = customer_parameter(
    customer_type      = 'static',
    num_customers      = [5,10],
    first_demand_step  = [0,0],
    demand_after_steps = None,
    demand_add         = 1,
    max_demand         = 10,
    init_value         = [1,1],
    signal_list        = [1,1,-1])


standard_depots = depot_parameter(
    num_depots       = 1,
    max_stock        = None,
    resupply_rate    = 1,
    unlimited_supply = True,
    init_value       = 30,
    signal_list      = [1,1,-1])


# Simulation Prameter:
# ----------------------------------------------------------------------------------------------------------------

all_parameter_list = [
    standard_MV_cargo, standard_range, street_travel,
    standard_UV_cargo, standard_battery, arial_travel,
    standard_customers, standard_depots
    ]


standard_simulation = simulation_parameter(
    grid            = [10,10],
    coord_type      = 'exact',
    locked_travel   = False,
    num_MV          = 2,
    num_UV_per_MV   = 2,
    prio_v          = 'UV', # 'MV'
    same_step_cargo = True,
    same_step_v     = True
    )


# Visulation Prameter:
# ----------------------------------------------------------------------------------------------------------------

standard_visual = visual_parameter(
    grid_surface_dim    = [400, 400],
    grid_padding        = 10,
    info_surface_height = 120,
    marker_size         = 6)


# Action Interpreter Prameter:
# ----------------------------------------------------------------------------------------------------------------

standard_actions = output_parameter(
    contin_outputs   = ['coord','amount','v_amount'], # ['load', 'unload', v_load, v_unload]
    discrete_outputs = ['nodes', 'v_to_load', 'move', 'load_unload', 'v_load_unload'],
    discrete_dims    = 20,
    combine          = 'contin', # 'discrete', 'by_categ', 'all', list of lists of output names
    )


# State Interpreter Prameter:
# ----------------------------------------------------------------------------------------------------------------

standard_states = input_parameter(
    image_input       = ['grid'],
    contin_inputs     = ['coordinates','values','vehicles','customers','depots'],
    discrete_inputs   = ['binary'],
    discrete_dims     = 20,
    combine_per_index = ['per_vehicle', 'per_customer', 'per_depot'], # list of input name lists
    combine_per_type  = None,
    # Flattens per combined (and all inputs not in a combined list),
    # if no combination are used everything will be flattened,
    flatten           = True,
    flatten_images    = False)


# Reward Calculator Prameter:
# ----------------------------------------------------------------------------------------------------------------

standard_rewards = reward_parameter(
    reward_modes        = None, #['normalized', 'discounted']
    reward_type         = 'single_vehicle', # 'multi_vehicle', 'sum_vehicle'
    restriction_rewards = ['battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
    action_rewards      = ['compare_coord','free_to_travel','unloading_v','free_to_unload_v','free_to_be_loaded_v','free_to_load_v','free_to_unload_cargo','free_to_load_cargo'])


# Init env:
env = CustomEnv('test',all_parameter_list,standard_simulation,standard_visual,standard_actions,standard_states,standard_rewards)
agent = Agent(env)

agent.test_agent(1)
