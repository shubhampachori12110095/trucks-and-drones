
from main.environment import CustomEnv
from simulation.simulation import simulation_parameter
from simulation.vehicles import battery_parameter, range_parameter, travel_parameter, cargo_parameter
from simulation.nodes import customer_parameter, depot_parameter
from simulation.action_interpreter import BaseActionInterpreter
from simulation.state_interpreter import BaseStateInterpreter
from simulation.reward_calculator import BaseRewardCalculator

from logger import TrainingLogger, TestingLogger


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
    cargo_weigth_UV    = 1,
    cargo_per_step     = 1,
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
    max_demand         = None,
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
    grid          = [10,10],
    coord_type    = 'exact',
    locked_travel = False,
    num_MV        = 2,
    num_UV_per_MV = 2)


# Visulation Prameter:
# ----------------------------------------------------------------------------------------------------------------

standard_visual = reward_parameter(
    grid_surface_dim    = [400, 400],
    grid_padding        = 10,
    info_surface_height = 120,
    marker_size         = 6)


# Action Interpreter Prameter:
# ----------------------------------------------------------------------------------------------------------------



# State Interpreter Prameter:
# ----------------------------------------------------------------------------------------------------------------

