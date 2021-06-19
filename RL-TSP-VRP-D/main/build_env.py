import numpy as np
import gym

from main.temp_database import BaseTempDatabase

from main.simulation.vehicles import BaseVehicleCreator
from main.simulation.nodes import BaseNodeCreator

from main.visualizer import BaseVisualizer
from main.state_interpreter import BaseStateInterpreter
from main.action_interpreter import BaseActionInterpreter
from main.reward_calculator import BaseRewardCalculator

from main.environment import CustomEnv


class BuildEnvironment:

    def __init__(self, 
            name: str, 
            grid: (list, tuple, np.ndarray) = [10,10],
            reward_signals: (list, tuple, np.ndarray) = [1,1,-1]
        ):

        self.name = name
        self.grid = grid
        self.reward_signals = reward_signals

        self.vehicle_params = []
        self.node_params = []

        self.visual_params = None
        self.obs_params = None
        self.act_params = None
        self.reward_params = None


    def vehicle(self,
            # number of vehicles:
            num: (int, list, tuple, np.ndarray) = 1,
            # vehicle name:
            v_name: str = 'vehicle', # alt: 'vehicle', 'truck', 'drone', 'robot'
            # loadable:
            loadable: bool = False,
            weight: (NoneType, int, list, tuple, np.ndarray) = 0,
            # range:
            range_type: str = 'simple', # alt: 'simple', 'battery'
            max_range: (NoneType, int, list, tuple, np.ndarray) = None,
            max_charge: (NoneType, int, list, tuple, np.ndarray) = None,
            init_charge: (str, NoneType, int, list, tuple, np.ndarray) = None,
            charge_rate: (str, NoneType, int, list, tuple, np.ndarray) = None,
            # travel:
            travel_type: str = 'street', # alt: 'street', arial
            speed: (int, list, tuple, np.ndarray) = 1,
            # cargo:
            cargo_type: str = 'standard', # alt: 'standard', 'standard+extra', 'standard+including'
            max_cargo: (NoneType, int, list, tuple, np.ndarray) = None,
            init_cargo: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            cargo_rate: (NoneType, int, list, tuple, np.ndarray) = None,
            # vehicle capacity:
            max_v_cap: (NoneType, int, list, tuple, np.ndarray) = 0,
            v_rate: (NoneType, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, NoneType) = 'circle', # 'triangle-up', 'triangle-down' 'rectangle'
            color: (str, NoneType, list, tuple, np.ndarray) = 'red',
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


    def truck(self,
            # number of trucks:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = False,
            weight: (NoneType, int, list, tuple, np.ndarray) = 0,
            # range:
            range_type: str = 'simple', # alt: 'simple', 'battery'
            max_range: (NoneType, int, list, tuple, np.ndarray) = None,
            max_charge: (NoneType, int, list, tuple, np.ndarray) = None,
            init_charge: (str, NoneType, int, list, tuple, np.ndarray) = None,
            charge_rate: (str, NoneType, int, list, tuple, np.ndarray) = None,
            # travel:
            travel_type: str = 'street', # alt: 'street', arial
            speed: (int, list, tuple, np.ndarray) = 1,
            # cargo:
            cargo_type: str = 'standard+extra', # alt: 'standard', 'standard+extra', 'standard+including'
            max_cargo: (NoneType, int, list, tuple, np.ndarray) = None,
            cargo_rate: (NoneType, int, list, tuple, np.ndarray) = None,
            init_cargo: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            # vehicle capacity:
            max_v_cap: (NoneType, int, list, tuple, np.ndarray) = 1,
            v_rate: (NoneType, int, list, tuple, np.ndarray) = 1,
            # visualization:
            symbol: (str, NoneType) = 'circle', # 'triangle-up', 'triangle-down' 'rectangle'
            color: (str, NoneType, list, tuple, np.ndarray) = 'purple',
        ):

        self.vehicle(num,'truck',loadable,weight,range_type,max_range,max_charge,init_charge,charge_rate,travel_type
            speed,cargo_type,max_cargo,init_cargo,cargo_rate,max_v_cap,v_rate, symbol, color)


    def drone(self,
            # number of drones:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = True,
            weight: (NoneType, int, list, tuple, np.ndarray) = 0,
            # range:
            range_type: str = 'battery', # alt: 'simple', 'battery'
            max_range: (NoneType, int, list, tuple, np.ndarray) = None,
            max_charge: (NoneType, int, list, tuple, np.ndarray) = 100,
            init_charge: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            charge_rate: (str, NoneType, int, list, tuple, np.ndarray) = None,
            # travel:
            travel_type: str = 'arial', # alt: 'street', arial
            speed: (int, list, tuple, np.ndarray) = 1,
            # cargo:
            cargo_type: str = 'standard', # alt: 'standard', 'standard+extra', 'standard+including'
            max_cargo: (NoneType, int, list, tuple, np.ndarray) = 1,
            cargo_rate: (NoneType, int, list, tuple, np.ndarray) = 1,
            init_cargo: (str, NoneType, int, list, tuple, np.ndarray) = 0,
            # vehicle capacity:
            max_v_cap: (NoneType, int, list, tuple, np.ndarray) = 0,
            v_rate: (NoneType, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, NoneType) = 'triangle-up', # 'triangle-up', 'triangle-down' 'rectangle'
            color: (str, NoneType, list, tuple, np.ndarray) = 'blue',
        ):

        self.vehicle(num,'drone',loadable,weight,range_type,max_range,max_charge,init_charge,charge_rate,travel_type
            speed,cargo_type,max_cargo,init_cargo,cargo_rate,max_v_cap,v_rate, symbol, color)


    def robot(self
            # number of robots:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = True,
            weight: (NoneType, int, list, tuple, np.ndarray) = 0,
            # range:
            range_type: str = 'battery', # alt: 'simple', 'battery'
            max_range: (NoneType, int, list, tuple, np.ndarray) = None,
            max_charge: (NoneType, int, list, tuple, np.ndarray) = 100,
            init_charge: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            charge_rate: (str, NoneType, int, list, tuple, np.ndarray) = None,
            # travel:
            travel_type: str = 'street', # alt: 'street', arial
            speed: (int, list, tuple, np.ndarray) = 0.5,
            # cargo:
            cargo_type: str = 'standard', # alt: 'standard', 'standard+extra', 'standard+including'
            max_cargo: (NoneType, int, list, tuple, np.ndarray) = 1,
            init_cargo: (str, NoneType, int, list, tuple, np.ndarray) = 0,
            cargo_rate: (NoneType, int, list, tuple, np.ndarray) = 1,
            # vehicle capacity:
            max_v_cap: (NoneType, int, list, tuple, np.ndarray) = 0,
            v_rate: (NoneType, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, NoneType) = 'triangle-down', # 'triangle-up', 'triangle-down' 'rectangle'
            color: (str, NoneType, int, list, tuple, np.ndarray) = 'light-blue',
        ):

        self.vehicle(num,'robot',loadable,weight,range_type,max_range,max_charge,init_charge,charge_rate,travel_type
            speed,cargo_type,max_cargo,init_cargo,cargo_rate,max_v_cap,v_rate, symbol, color)


    def node(self
            # number of nodes:
            num: (int, list, tuple, np.ndarray) = 1,
            # node name:
            n_name: str = 'node', # alt: 'node', 'depot', 'customer'
            # items (stock if node is depot and demand if node is customer):
            max_items: (NoneType, int, list, tuple, np.ndarray) = 10,
            init_items: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            item_rate: (NoneType, int, list, tuple, np.ndarray) = None,
            item_recharge: (NoneType, int, list, tuple, np.ndarray) = 0,
            init_items_at_step: (NoneType, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, NoneType) = 'rectangle', # 'triangle-up', 'triangle-down', 'rectangle'
            color: (str, NoneType, list, tuple, np.ndarray) = 'orange',
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


    def depot(self
            # number of nodes:
            num: (int, list, tuple, np.ndarray) = 1,
            # items (stock if node is depot and demand if node is customer):
            max_items: (NoneType, int, list, tuple, np.ndarray) = 10,
            init_items: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            item_rate: (NoneType, int, list, tuple, np.ndarray) = None,
            item_recharge: (NoneType, int, list, tuple, np.ndarray) = 0,
            init_items_at_step: (NoneType, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, NoneType) = 'rectangle', # 'triangle-up', 'triangle-down', 'rectangle'
            color: (str, NoneType, list, tuple, np.ndarray) = 'orange',
        ):

        self.node(num,'depot',max_items,init_items,item_rate,item_recharge,init_items_at_step)


    def customer(self
            # number of nodes:
            num: (int, list, tuple, np.ndarray) = 1,
            # items (stock if node is depot and demand if node is customer):
            max_items: (NoneType, int, list, tuple, np.ndarray) = 10,
            init_items: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            item_rate: (NoneType, int, list, tuple, np.ndarray) = None,
            item_recharge: (NoneType, int, list, tuple, np.ndarray) = 0,
            init_items_at_step: (NoneType, int, list, tuple, np.ndarray) = 0,
            # visualization:
            symbol: (str, NoneType) = 'rectangle', # 'triangle-up', 'triangle-down', 'rectangle'
            color: (str, NoneType, list, tuple, np.ndarray) = 'yellow',
        ):

        self.node(num,'customer',max_items,init_items,item_rate,item_recharge,init_items_at_step)


    def visual(self,
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


    def observations(self,
            image_input: (NoneType, list, tuple, np.ndarray) = ['grid'],
            contin_inputs: (NoneType, list, tuple, np.ndarray) = ['coordinates','values','vehicles','customers','depots'],
            discrete_inputs: (NoneType, list, tuple, np.ndarray) = ['binary'],
            discrete_bins: int = 20, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            combine_per_index: (NoneType, list, tuple, np.ndarray) = ['per_vehicle', 'per_customer', 'per_depot'], # list of input name lists
            combine_per_type: (NoneType, list, tuple, np.ndarray) = None,
            # Flattens per combined (and all inputs not in a combined list),
            # if no combination are used everything will be flattened,
            flatten: bool = True,
            flatten_images: bool = False
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
        }


    def dummy_observations(self,
            image_input: (NoneType, list, tuple, np.ndarray) = None,
            contin_inputs: (NoneType, list, tuple, np.ndarray) = None,
            discrete_inputs: (NoneType, list, tuple, np.ndarray) = None,
            discrete_bins: int = 20, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            combine_per_index: (NoneType, list, tuple, np.ndarray) = None, # list of input name lists
            combine_per_type: (NoneType, list, tuple, np.ndarray) = None,
            # Flattens per combined (and all inputs not in a combined list),
            # if no combination are used everything will be flattened,
            flatten: bool = False,
            flatten_images: bool = False
        ):

        self.observations(image_input,contin_inputs,discrete_inputs,discrete_bins,combine_per_index,
            combine_per_type,flatten,flatten_images)


    def actions(self,
            mode: str = 'single_vehicle', # 'multi_vehicle'
            flattened: str = 'per_output', #'per_vehicle', #'all'
            contin_outputs: (NoneType, list, tuple, np.ndarray) = ['coord','amount','v_amount'],
            discrete_outputs: (NoneType, list, tuple, np.ndarray) = ['nodes', 'v_to_load'],
            binary_discrete: (NoneType, list, tuple, np.ndarray) = ['move', 'load_unload', 'v_load_unload'],
            binary_contin: (NoneType, list, tuple, np.ndarray) = [],
            discrete_bins: int = 20,
            combine: (str, NoneType, list, tuple, np.ndarray) = 'contin', # 'discrete', 'by_categ', 'all', list of lists of output names
        ):

        self.act_params = {
            'mode': mode,
            'flattened': flattened,
            'contin_outputs': contin_outputs,
            'discrete_outputs': discrete_outputs,
            'binary_discrete': binary_discrete,
            'binary_contin': binary_contin,
            'discrete_bins': discrete_bins,
            'combine': combine,
        }


    def dummy_actions(self,
            mode: str = 'single_vehicle', # 'multi_vehicle'
            flattened: str = 'per_output', #'per_vehicle', #'all'
            contin_outputs: (NoneType, list, tuple, np.ndarray) = None,
            discrete_outputs: (NoneType, list, tuple, np.ndarray) = None,
            binary_discrete: (NoneType, list, tuple, np.ndarray) = None,
            binary_contin: (NoneType, list, tuple, np.ndarray) = None,
            discrete_bins: int = 20,
            combine: (str, NoneType, list, tuple, np.ndarray) = None, # 'discrete', 'by_categ', 'all', list of lists of output names
        ):

        self.actions(mode,flattened,contin_outputs,discrete_outputs,
            binary_discrete,binary_contin,discrete_bins,combine)


    def rewards(self,
            reward_modes: (str, NoneType) = None, #['normalized', 'discounted']
            reward_type: str = 'single_vehicle', # 'multi_vehicle', 'sum_vehicle'
            restriction_rewards: (NoneType, list, tuple, np.ndarray) = ['battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
            action_rewards: (NoneType, list, tuple, np.ndarray) = ['compare_coord','free_to_travel','unloading_v','free_to_unload_v','free_to_be_loaded_v','free_to_load_v','free_to_unload_cargo','free_to_load_cargo']
        ):

        self.reward_params = {
            'reward_modes': reward_modes,
            'reward_type': reward_type,
            'restriction_rewards': restriction_rewards,
            'action_rewards': action_rewards,
        }


    def compile(self,
            TempDatabase: BaseTempDatabase = BaseTempDatabase,
            VehicleCreator: BaseVehicleCreator = BaseVehicleCreator,
            NodeCreator: BaseNodeCreator = BaseNodeCreator,
            Simulator: BaseSimulator = BaseSimulator,
            Visualizer: BaseVisualizer = BaseVisualizer,
            ObsEncoder: BaseObsEncoder = BaseObsEncoder,
            ActDecoder: BaseActDecoder = BaseActDecoder,
            RewardCalculator: BaseRewardCalculator = BaseRewardCalculator,
        ):

        # Check vehicle parameter:
        if len(self.vehicle_params) == 0:
            self.vehicle()
            print('Using standard vehicle parameter')

        # Check node parameter:
        if len(self.node_params) == 0:
            self.depot()
            print('Using standard depot parameter')
            self.customer()
            print('Using standard customer parameter')
        else:
            node_types = [node['n_type'] for node in self.node_params]
            # Check depot parameter:
            if 'depot' not in node_types:
                self.depot()
                print('Using standard depot parameter')
            # Check customer parameter:
            if 'customer' not in node_types:
                self.customer()
                print('Using standard customer parameter')

        # Check visual parameter:
        if self.visual_params is None:
            self.visual()
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
        self.temp_db = TempDatabase(self.name, self.grid, self.reward_signals)

        # Init vehicle and node creators:
        self.vehicle_creator = VehicleCreator(self.vehicle_params, self.temp_db)
        self.node_creator = NodeCreator(self.node_params, self.temp_db)

        # Init simulation:
        self.simulation = Simulator(self.temp_db, self.vehicle_creator, self.node_creator)
        
        # Init visualization:
        self.visualizor = Visualizer(self.visual_params, self.temp_db)

        # Init observation and actions encoding/decoding:
        self.obs_encoder = ObsEncoder(self.obs_params, self.temp_db, self.visualizor)
        self.act_decoder = ActDecoder(self.act_params, self.temp_db, self.simulation)

        # Init reward calculations:
        self.reward_calc = RewardCalculator(self.reward_params, self.temp_db)


    def build(self) -> gym.Env:

        return CustomEnv(
            self.name, self.simulation, self.visualizor, self.obs_encoder, self.act_decoder, self.reward_calc
        )