import numpy as np
import gym

from trucks_and_drones.simulation.temp_database import BaseTempDatabase
from trucks_and_drones.simulation.vehicles import BaseVehicleCreator
from trucks_and_drones.simulation.nodes import BaseNodeCreator
from trucks_and_drones.simulation.auto_agent import BaseAutoAgent
from trucks_and_drones.simulation.simulation import BaseSimulator

from trucks_and_drones.visualizer import BaseVisualizer
from trucks_and_drones.simulation.state_interpreter import BaseObsEncoder
from trucks_and_drones.simulation.action_interpreter import BaseActDecoder
from trucks_and_drones.reward_calculator import BaseRewardCalculator

from trucks_and_drones.environment import CustomEnv


class BuildEnvironment:

    def __init__(
            self,
            name: str, 
            grid: (list, tuple, np.ndarray) = [10, 10],
            reward_signals: (list, tuple, np.ndarray) = [1, 1, -1],
            max_steps_per_episode: int = 1000,
            debug_mode: bool = False,
    ):

        self.name = name
        self.grid = grid
        self.reward_signals = reward_signals
        self.max_steps_per_episode = max_steps_per_episode
        self.debug_mode = debug_mode

        self.vehicle_params = []
        self.node_params = []

        self.visual_params = None
        self.obs_params = None
        self.act_params = None
        self.reward_params = None

    def vehicles(
            self,
            # number of vehicles:
            num: (int, list, tuple, np.ndarray) = 1,
            # vehicle name:
            v_name: str = 'vehicle',  # alt: 'vehicle', 'truck', 'drone', 'robot'
            # loadable:
            loadable: bool = False,
            weight: (None, int, list, tuple, np.ndarray) = 0,
            # range:
            range_type: str = 'simple',  # alt: 'simple', 'battery'
            max_range: (None, int, list, tuple, np.ndarray) = None,
            max_charge: (None, int, list, tuple, np.ndarray) = None,
            init_charge: (str, None, int, list, tuple, np.ndarray) = None,
            charge_rate: (str, None, int, list, tuple, np.ndarray) = None,
            # travel:
            travel_type: str = 'street',  # alt: 'street', arial
            speed: (float, int, list, tuple, np.ndarray) = 1,
            # cargo:
            cargo_type: str = 'standard',  # alt: 'standard', 'standard+extra', 'standard+including'
            max_cargo: (None, int, list, tuple, np.ndarray) = None,
            init_cargo: (str, None, int, list, tuple, np.ndarray) = 0,
            cargo_rate: (None, int, list, tuple, np.ndarray) = None,
            # vehicle capacity:
            max_v_cap: (None, int, list, tuple, np.ndarray) = 0,
            v_rate: (None, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, None) = 'circle',  # 'triangle-up', 'triangle-down' 'rectangle'
            color: (str, None, list, tuple, np.ndarray) = 'red',
    ):

        self.vehicle_params.append({
            'num': num,
            'v_name': v_name,
            'loadable': loadable,
            'weight': weight,
            'range_type': range_type,
            'max_range': max_range,
            'max_charge': max_charge,
            'init_charge': init_charge,
            'charge_rate': charge_rate,
            'travel_type': travel_type,
            'speed': speed,
            'cargo_type': cargo_type,
            'max_cargo': max_cargo,
            'init_cargo': init_cargo,
            'cargo_rate': cargo_rate,
            'max_v_cap': max_v_cap,
            'v_rate': v_rate,
            'symbol': symbol,
            'color': color,
            }
        )

    def trucks(
            self,
            # number of trucks:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = False,
            weight: (None, int, list, tuple, np.ndarray) = 0,
            # range:
            range_type: str = 'simple',  # alt: 'simple', 'battery'
            max_range: (None, int, list, tuple, np.ndarray) = None,
            max_charge: (None, int, list, tuple, np.ndarray) = None,
            init_charge: (str, None, int, list, tuple, np.ndarray) = None,
            charge_rate: (str, None, int, list, tuple, np.ndarray) = None,
            # travel:
            travel_type: str = 'street',  # alt: 'street', arial
            speed: (float, int, list, tuple, np.ndarray) = 1,
            # cargo:
            cargo_type: str = 'standard+extra',  # alt: 'standard', 'standard+extra', 'standard+including'
            max_cargo: (None, int, list, tuple, np.ndarray) = None,
            cargo_rate: (None, int, list, tuple, np.ndarray) = None,
            init_cargo: (str, None, int, list, tuple, np.ndarray) = 0,
            # vehicle capacity:
            max_v_cap: (None, int, list, tuple, np.ndarray) = 1,
            v_rate: (None, int, list, tuple, np.ndarray) = None,
            # visualization:
            symbol: (str, None) = 'circle',  # 'triangle-up', 'triangle-down' 'rectangle'
            color: (str, None, list, tuple, np.ndarray) = 'purple',
    ):

        self.vehicles(num, 'truck', loadable, weight, range_type, max_range, max_charge, init_charge,
                      charge_rate, travel_type, speed, cargo_type, max_cargo, init_cargo, cargo_rate,
                      max_v_cap, v_rate, symbol, color)

    def drones(
            self,
            # number of drones:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = True,
            weight: (None, int, list, tuple, np.ndarray) = 0,
            # range:
            range_type: str = 'simple',  # alt: 'simple', 'battery'
            max_range: (None, int, list, tuple, np.ndarray) = 4,
            max_charge: (None, int, list, tuple, np.ndarray) = 100,
            init_charge: (str, None, int, list, tuple, np.ndarray) = 0,
            charge_rate: (str, None, int, list, tuple, np.ndarray) = None,
            # travel:
            travel_type: str = 'arial',  # alt: 'street', arial
            speed: (float, int, list, tuple, np.ndarray) = 1,
            # cargo:
            cargo_type: str = 'standard',  # alt: 'standard', 'standard+extra', 'standard+including'
            max_cargo: (None, int, list, tuple, np.ndarray) = 1,
            cargo_rate: (None, int, list, tuple, np.ndarray) = None,
            init_cargo: (str, None, int, list, tuple, np.ndarray) = 0,
            # vehicle capacity:
            max_v_cap: (None, int, list, tuple, np.ndarray) = 0,
            v_rate: (None, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, None) = 'triangle-up',  # 'triangle-up', 'triangle-down' 'rectangle'
            color: (str, None, list, tuple, np.ndarray) = 'blue',
    ):

        self.vehicles(num, 'drone', loadable, weight, range_type, max_range, max_charge, init_charge, charge_rate,
                      travel_type, speed, cargo_type, max_cargo, init_cargo, cargo_rate, max_v_cap, v_rate,
                      symbol, color)

    def robots(
            self,
            # number of robots:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = True,
            weight: (None, int, list, tuple, np.ndarray) = 0,
            # range:
            range_type: str = 'simple',  # alt: 'simple', 'battery'
            max_range: (None, int, list, tuple, np.ndarray) = 4,
            max_charge: (None, int, list, tuple, np.ndarray) = 100,
            init_charge: (str, None, int, list, tuple, np.ndarray) = 0,
            charge_rate: (str, None, int, list, tuple, np.ndarray) = None,
            # travel:
            travel_type: str = 'street',  # alt: 'street', arial
            speed: (float, int, list, tuple, np.ndarray) = 0.5,
            # cargo:
            cargo_type: str = 'standard',  # alt: 'standard', 'standard+extra', 'standard+including'
            max_cargo: (None, int, list, tuple, np.ndarray) = 1,
            init_cargo: (str, None, int, list, tuple, np.ndarray) = 0,
            cargo_rate: (None, int, list, tuple, np.ndarray) = None,
            # vehicle capacity:
            max_v_cap: (None, int, list, tuple, np.ndarray) = 0,
            v_rate: (None, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, None) = 'triangle-down',  # 'triangle-up', 'triangle-down' 'rectangle'
            color: (str, None, int, list, tuple, np.ndarray) = 'light-blue',
        ):

        self.vehicles(num, 'robot', loadable, weight, range_type, max_range, max_charge, init_charge, charge_rate,
                      travel_type, speed, cargo_type, max_cargo, init_cargo, cargo_rate, max_v_cap, v_rate,
                      symbol, color)

    def nodes(
            self,
            # number of nodes:
            num: (int, list, tuple, np.ndarray) = 1,
            # node name:
            n_name: str = 'node',  # alt: 'node', 'depot', 'customer'
            # items (stock if node is depot and demand if node is customer):
            max_items: (None, int, list, tuple, np.ndarray) = None,
            init_items: (str, None, int, list, tuple, np.ndarray) = None,
            item_rate: (None, int, list, tuple, np.ndarray) = None,
            item_recharge: (None, int, list, tuple, np.ndarray) = 0,
            init_items_at_step: (None, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, None) = 'rectangle',  # 'triangle-up', 'triangle-down', 'rectangle'
            color: (str, None, list, tuple, np.ndarray) = 'orange',
        ):

        self.node_params.append({
            'num': num,
            'n_name': n_name,
            'max_items': max_items,
            'init_items': init_items,
            'item_rate': item_rate,
            'item_recharge': item_recharge,
            'init_items_at_step': init_items_at_step,
            'symbol': symbol, 
            'color': color,
            }
        )

    def depots(
            self,
            # number of nodes:
            num: (int, list, tuple, np.ndarray) = 1,
            # items (stock if node is depot and demand if node is customer):
            max_items: (None, int, list, tuple, np.ndarray) = None,
            init_items: (str, None, int, list, tuple, np.ndarray) = None,
            item_rate: (None, int, list, tuple, np.ndarray) = None,
            item_recharge: (None, int, list, tuple, np.ndarray) = 0,
            init_items_at_step: (None, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, None) = 'rectangle', # 'triangle-up', 'triangle-down', 'rectangle'
            color: (str, None, list, tuple, np.ndarray) = 'orange',
        ):

        self.nodes(num,'depot',max_items,init_items,item_rate,item_recharge,init_items_at_step,symbol,color)

    def customers(
            self,
            # number of nodes:
            num: (int, list, tuple, np.ndarray) = 1,
            # items (stock if node is depot and demand if node is customer):
            max_items: (None, int, list, tuple, np.ndarray) = 1,
            init_items: (str, None, int, list, tuple, np.ndarray) = 'max',
            item_rate: (None, int, list, tuple, np.ndarray) = None,
            item_recharge: (None, int, list, tuple, np.ndarray) = 0,
            init_items_at_step: (None, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, None) = 'rectangle', # 'triangle-up', 'triangle-down', 'rectangle'
            color: (str, None, list, tuple, np.ndarray) = 'light-grey',
        ):

        self.nodes(num,'customer',max_items,init_items,item_rate,item_recharge,init_items_at_step,symbol,color)

    def visuals(
            self,
            grid_surface_dim: (list, tuple, np.ndarray) = [600, 600],
            grid_padding: int = 20,
            info_surface_height: int = 240,
            marker_size: int = 20
        ):

        self.visual_params = {
            'grid_surface_dim': grid_surface_dim,
            'grid_padding': grid_padding,
            'info_surface_height': info_surface_height,
            'marker_size': marker_size,
        }

    def observations(
            self,
            image_input: (None, list, tuple, np.ndarray) = [],
            contin_inputs: (None, list, tuple, np.ndarray) = None, #['values'],
            discrete_inputs: (None, list, tuple, np.ndarray) = None, #['binary', 'coordinates'],
            discrete_bins: int = 20, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            combine_per_index: (None, list, tuple, np.ndarray) = ['per_vehicle', 'per_customer', 'per_depot'], # list of input name lists
            combine_per_type: (None, list, tuple, np.ndarray) = None,
            # Flattens per combined (and all inputs not in a combined list),
            # if no combination are used everything will be flattened,
            flatten: bool = True,
            flatten_images: bool = False,
            output_as_array: bool = True,
        ):
        
        self.obs_params = {
            'image_input': image_input,
            'contin_inputs': contin_inputs,
            'discrete_inputs': discrete_inputs,
            'discrete_bins': discrete_bins,
            'combine_per_index': combine_per_index,
            'combine_per_type': combine_per_type,
            'flatten': flatten,
            'flatten_images': flatten_images,
            'output_as_array': output_as_array,
        }

    def dummy_observations(
            self,
            image_input: (None, list, tuple, np.ndarray) = None,
            contin_inputs: (None, list, tuple, np.ndarray) = None,
            discrete_inputs: (None, list, tuple, np.ndarray) = None,
            discrete_bins: int = 20, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            combine_per_index: (None, list, tuple, np.ndarray) = None, # list of input name lists
            combine_per_type: (None, list, tuple, np.ndarray) = None,
            # Flattens per combined (and all inputs not in a combined list),
            # if no combination are used everything will be flattened,
            flatten: bool = False,
            flatten_images: bool = False
        ):

        self.observations(image_input,contin_inputs,discrete_inputs,discrete_bins,combine_per_index,
            combine_per_type,flatten,flatten_images)

    def actions(
            self,
            mode: str = 'single_vehicle', # 'multi_vehicle'
            flattened: str = 'per_output', #'per_vehicle', #'all'
            contin_outputs: (None, list, tuple, np.ndarray) = [],#['coord','amount','v_amount'],
            discrete_outputs: (None, list, tuple, np.ndarray) = [], #['nodes', 'v_to_load'],
            binary_discrete: (None, list, tuple, np.ndarray) = [], #['move', 'load_unload', 'v_load_unload'],
            binary_contin: (None, list, tuple, np.ndarray) = [],
            num_discrete_bins: int = 20,
            combine: (str, None, list, tuple, np.ndarray) = 'contin', # 'discrete', 'by_categ', 'all', list of lists of output names
            multiple_action_spaces: bool = False,
        ):

        self.act_params = {
            'mode': mode,
            'flattened': flattened,
            'contin_outputs': contin_outputs,
            'discrete_outputs': discrete_outputs,
            'binary_discrete': binary_discrete,
            'binary_contin': binary_contin,
            'num_discrete_bins': num_discrete_bins,
            'combine': combine,
            'multiple_action_spaces': multiple_action_spaces,
        }

    def dummy_actions(
            self,
            mode: str = 'single_vehicle', # 'multi_vehicle'
            flattened: str = 'per_output', #'per_vehicle', #'all'
            contin_outputs: (None, list, tuple, np.ndarray) = [],
            discrete_outputs: (None, list, tuple, np.ndarray) = [],
            binary_discrete: (None, list, tuple, np.ndarray) = [],
            binary_contin: (None, list, tuple, np.ndarray) = [],
            discrete_bins: int = 20,
            combine: (str, None, list, tuple, np.ndarray) = None, # 'discrete', 'by_categ', 'all', list of lists of output names
        ):

        self.actions(mode,flattened,contin_outputs,discrete_outputs,
            binary_discrete,binary_contin,discrete_bins,combine)

    def rewards(
            self,
            reward_modes: (str, None) = None, #['normalized', 'discounted']
            reward_type: str = 'single_vehicle', # 'multi_vehicle', 'sum_vehicle'
            restriction_rewards: (None, list, tuple, np.ndarray) = ['battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
            action_rewards: (None, list, tuple, np.ndarray) = ['compare_coord','free_to_travel','unloading_v','free_to_unload_v','free_to_be_loaded_v','free_to_load_v','free_to_unload_cargo','free_to_load_cargo']
        ):

        self.reward_params = {
            'reward_modes': reward_modes,
            'reward_type': reward_type,
            'restriction_rewards': restriction_rewards,
            'action_rewards': action_rewards,
        }

    def compile(
            self,
            TempDatabase: BaseTempDatabase = BaseTempDatabase,
            VehicleCreator: BaseVehicleCreator = BaseVehicleCreator,
            NodeCreator: BaseNodeCreator = BaseNodeCreator,
            AutoAgent: BaseAutoAgent = BaseAutoAgent,
            Simulator: BaseSimulator = BaseSimulator,
            Visualizer: BaseVisualizer = BaseVisualizer,
            ObsEncoder: BaseObsEncoder = BaseObsEncoder,
            ActDecoder: BaseActDecoder = BaseActDecoder,
            RewardCalculator: BaseRewardCalculator = BaseRewardCalculator,
        ):

        # Check vehicle parameter:
        if len(self.vehicle_params) == 0:
            self.vehicles()
            print('Using standard vehicle parameter')

        # Check node parameter:
        if len(self.node_params) == 0:
            self.depots()
            print('Using standard depot parameter')
            self.customers()
            print('Using standard customer parameter')
        else:
            node_types = [node['n_name'] for node in self.node_params]
            # Check depot parameter:
            if 'depot' not in node_types:
                self.depots()
                print('Using standard depot parameter')
            # Check customer parameter:
            if 'customer' not in node_types:
                self.customers()
                print('Using standard customer parameter')

        # Check visual parameter:
        if self.visual_params is None:
            self.visuals()
            print('Using standard visual parameter')

        # Check observations parameter:
        if self.obs_params is None:
            self.observations()
            print('Using standard observations parameter')

        # Check actions parameter:
        if self.act_params is None:
            self.actions()
            print('Using standard actions parameter')

        # Check rewards parameter:
        if self.reward_params is None:
            self.rewards()
            print('Using standard rewards parameter')

        # Init temporary database:
        self.temp_db = TempDatabase(self.name, self.grid, self.reward_signals, self.debug_mode)

        # Init vehicle and node creators:
        self.node_creator = NodeCreator(self.node_params, self.temp_db)
        self.vehicle_creator = VehicleCreator(self.vehicle_params, self.temp_db)
        self.auto_agent = AutoAgent(self.temp_db)

        # Init simulation:
        self.simulation = Simulator(self.temp_db, self.vehicle_creator, self.node_creator, self.auto_agent)
        
        # Init visualization:
        self.visualizor = Visualizer(self.name, self.visual_params, self.temp_db)

        # Init observation and actions encoding/decoding:
        from trucks_and_drones.simulation import acts
        from trucks_and_drones.simulation import obs
        #self.obs_encoder = ObsEncoder(self.obs_params, self.temp_db, self.visualizor)
        self.obs_encoder = obs.SimpleObs(self.temp_db)
        #self.act_decoder = ActDecoder(self.act_params, self.temp_db, self.simulation)
        self.act_decoder = acts.TSPDroneAction(self.temp_db, self.simulation)

        # Init reward calculations:
        self.reward_calc = RewardCalculator(self.reward_params, self.temp_db)

    def build(self) -> gym.Env:

        return CustomEnv(
            self.name,
            self.simulation, self.visualizor, self.obs_encoder, self.act_decoder, self.reward_calc
        )
