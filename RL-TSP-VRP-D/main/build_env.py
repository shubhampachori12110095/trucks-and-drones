import numpy as np

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


    def vehicle(self,
            # number of vehicles:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = False,
            # range or battery:
            range_type: str = 'range', # alt: 'range', 'battery'
            max_range: (NoneType, int, list, tuple, np.ndarray) = None,
            max_charge: (NoneType, int, list, tuple, np.ndarray) = None,
            init_charge: (str, NoneType, int, list, tuple, np.ndarray) = None,
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
        ):


    def truck(self,
            # number of trucks:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = False,
            # range or battery:
            range_type: str = 'range', # alt: 'range', 'battery'
            max_range: (NoneType, int, list, tuple, np.ndarray) = None,
            max_charge: (NoneType, int, list, tuple, np.ndarray) = None,
            init_charge: (str, NoneType, int, list, tuple, np.ndarray) = None,
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
        ):

        self.vehicle(num,loadable,range_type,max_range,max_charge,init_charge,travel_type
            speed,cargo_type,max_cargo,init_cargo,cargo_rate,max_v_cap,v_rate)


    def drone(self,
            # number of drones:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = True,
            # range or battery:
            range_type: str = 'battery', # alt: 'range', 'battery'
            max_range: (NoneType, int, list, tuple, np.ndarray) = None,
            max_charge: (NoneType, int, list, tuple, np.ndarray) = 100,
            init_charge: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
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
        ):

        self.vehicle(num,loadable,range_type,max_range,max_charge,init_charge,travel_type
            speed,cargo_type,max_cargo,init_cargo,cargo_rate,max_v_cap,v_rate)


    def robot(self
            # number of robots:
            num: (int, list, tuple, np.ndarray) = 1,
            # loadable:
            loadable: bool = True,
            # range or battery:
            range_type: str = 'battery', # alt: 'range', 'battery'
            max_range: (NoneType, int, list, tuple, np.ndarray) = None,
            max_charge: (NoneType, int, list, tuple, np.ndarray) = 100,
            init_charge: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
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
        ):

        self.vehicle(num,loadable,range_type,max_range,max_charge,init_charge,travel_type
            speed,cargo_type,max_cargo,init_cargo,cargo_rate,max_v_cap,v_rate)


    def node(self
            # number of nodes:
            num: (int, list, tuple, np.ndarray) = 1,
            # node type:
            n_type: str = 'depot', # alt: 'depot', 'customer'
            # items (stock if node is depot and demand if node is customer):
            max_items: (NoneType, int, list, tuple, np.ndarray) = 10,
            init_items: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            item_rate: (NoneType, int, list, tuple, np.ndarray) = None,
            item_recharge: (NoneType, int, list, tuple, np.ndarray) = 0,
            init_items_at_step: (NoneType, int, list, tuple, np.ndarray) = 0,
        ):


    def depot(self
            # number of nodes:
            num: (int, list, tuple, np.ndarray) = 1,
            # items (stock if node is depot and demand if node is customer):
            max_items: (NoneType, int, list, tuple, np.ndarray) = 10,
            init_items: (str, NoneType, int, list, tuple, np.ndarray) = 'max',
            item_rate: (NoneType, int, list, tuple, np.ndarray) = None,
            item_recharge: (NoneType, int, list, tuple, np.ndarray) = 0,
            init_items_at_step: (NoneType, int, list, tuple, np.ndarray) = 0,
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
        ):

        self.node(num,'customer',max_items,init_items,item_rate,item_recharge,init_items_at_step)


    def visual(self,
            grid_surface_dim: (list, tuple, np.ndarray) = [600, 600],
            grid_padding: int = 20,
            info_surface_height: int = 240,
            marker_size: int = 20
        ):


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


    def states(self):


    def rewards(self):


    def compile(self) -> gym.Env: