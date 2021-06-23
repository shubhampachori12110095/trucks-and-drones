'''
- euclidean route
- non-euclidean route

Abbreviations:

MV: Manned Vehicle
UV: Unmanned Vehicle
SoC: State of Charge


unterscheidung ob dinge gleichzeitig oder nacheinander passieren können:
- travel and unload/load
- unload UV and unload cargo for UV
'''
import numpy as np

from main.simulation.common_sim_func import param_interpret, l_ignore_none


# Base Simulator Class:
# ----------------------------------------------------------------------------------------------------------------

class BaseSimulator:

    def __init__(self, temp_db, vehicle_creator, node_creator, auto_agent):

        self.temp_db = temp_db
        self.vehicle_creator = vehicle_creator
        self.node_creator = node_creator
        self.auto_agent = auto_agent


    def reset_simulation(self):

        self.temp_db.init_db()
        self.node_creator.create()
        self.vehicle_creator.create()
        self.temp_db.reset_db()
        self.reset_round()


    def reset_round(self):
        self.v_count = 0
        self.v_indices = np.squeeze(np.argwhere(np.isnan(self.temp_db.time_till_fin)))#np.where( == np.nan)
        self.num_v = self.v_indices.size
        if self.num_v == 0:
            self.finish_step()
        elif self.num_v == 1:
            self.temp_db.cur_v_index = self.v_indices
        else:
            self.temp_db.cur_v_index = self.v_indices[self.v_count]


    def set_destination(self, coordinates=None):

        if coordinates is None:
            coordinates = self.auto_agent.find_destination()

        if coordinates is not None:
            self.temp_db.status_dict['v_dest'][self.temp_db.cur_v_index] = np.array(coordinates)
            self.temp_db.actions_list[self.temp_db.cur_v_index].append(['move', None, None])
            #print('new destination:', coordinates, 'for', self.temp_db.cur_v_index)

        #### hier c_waiting!


    def unload_vehicle(self, v_j=None, amount=None):

        if v_j is None:
            v_j = self.auto_agent.find_v_to_unload()

        if v_j is not None:
            self.temp_db.actions_list[self.temp_db.cur_v_index].append(['unload_v', v_j, amount])
            #print(v_j, 'to unload from', self.temp_db.cur_v_index, 'with', amount, 'items')


    def load_vehicle(self, v_j=None):

        if v_j is None:
            v_j = self.auto_agent.find_v_to_load()

        if v_j is not None:
            if self.temp_db.same_coord(self.temp_db.status_dict['v_coord'][v_j]):
                self.temp_db.actions_list[self.temp_db.cur_v_index].append(['load_v', v_j, None])
                #print(v_j, 'to load to', self.temp_db.cur_v_index)


    def unload_items(self, n_j=None, amount=None):

        if n_j is None:
            n_j = self.auto_agent.find_customer()

        if n_j is not None:
            if self.temp_db.same_coord(self.temp_db.status_dict['n_coord'][n_j]):
                self.temp_db.actions_list[self.temp_db.cur_v_index].append(['unload_i', n_j, amount])
                #print(amount, 'items to unload from', self.temp_db.cur_v_index, 'to', n_j)


    def load_items(self, n_j=None, amount=None):

        if n_j is None:
            n_j = self.auto_agent.find_depot()

        if n_j is not None:
            if self.temp_db.same_coord(self.temp_db.status_dict['n_coord'][n_j]):
                self.temp_db.actions_list[self.temp_db.cur_v_index].append(['load_i', n_j, amount])
                #print(amount, 'items to load to', self.temp_db.cur_v_index, 'from', n_j)


    def recharge_range(self, vehicle_i):

        if any(self.temp_db.status_dict['v_coord'][vehicle_i] for elem in recharge_coord):
            recharge_v(self.temp_db.base_groups[vehicle_i])


    def finish_step(self):

        if self.temp_db.terminal_state():
            return True

        self.v_count += 1
        
        if self.v_count >= self.num_v:
            self.actions_during_timeframe()
        else:
            self.temp_db.cur_v_index = self.v_indices[self.v_count]
        
        return False

    def actions_during_timeframe(self):

        [v.take_action(calc_time=True) for v in self.temp_db.base_groups['vehicles']]
        for key in self.temp_db.restr_dict.keys(): [restr.in_time() for restr in self.temp_db.restr_dict[key] if restr is not None]
        
        min_masked_array = np.nanmin(self.temp_db.time_till_fin)
        print('min_masked_array', min_masked_array)
        if not np.isnan(min_masked_array):
            self.temp_db.cur_time_frame = min_masked_array
        else:
            self.temp_db.cur_time_frame = 0
        print(self.temp_db.cur_time_frame)
        for key in self.temp_db.restr_dict.keys(): [restr.in_time() for restr in self.temp_db.restr_dict[key] if restr is not None]
        [v.take_action() for v in self.temp_db.base_groups['vehicles']]

        print(self.temp_db.actions_list)
        self.reset_round()
