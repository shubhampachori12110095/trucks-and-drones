import numpy as np


class BaseAutoAgent:

    def __init__(self, temp_db):

        self.temp_db = temp_db


    def find_destination(self):

        if (self.temp_db.base_groups['vehicles'][self.temp_db.cur_v_index].v_items.cur_value() > 0
            and 0 in self.temp_db.customers(self.temp_db.status_dict['n_waiting'])[0]
            and np.sum(self.temp_db.customers(self.temp_db.status_dict['n_items'])[0]) != 0):
            n_index = self.temp_db.nearest_neighbour(self.temp_db.customers(
                self.temp_db.status_dict['n_coord'],
                include=[[self.temp_db.status_dict['n_waiting'], 0]],
                exclude=[[self.temp_db.status_dict['n_items'], 0]]
            )
        )

        # self.temp_db.status_dict['n_waiting'][n_index] = 1

        else:
            n_index = self.find_depot()

        self.temp_db.status_dict['v_to_n'][self.temp_db.cur_v_index] = n_index
        if n_index is None:
            return None

        return self.temp_db.status_dict['n_coord'][n_index]


    def find_v_to_unload(self):
        
        if any(self.temp_db.v_transporting_v[self.temp_db.cur_v_index]):
            return self.temp_db.v_transporting_v[self.temp_db.cur_v_index][0]
        return None
    

    def find_v_to_load(self):

        if self.temp_db.constants_dict['v_is_truck'][self.temp_db.cur_v_index]:
            return self.temp_db.nearest_neighbour(self.temp_db.vehicles(
                    self.temp_db.status_dict['v_coord'],
                    include=[[self.temp_db.constants_dict['v_loadable'], 1], [self.temp_db.status_dict['v_free'], 1]]
                )
            )
        else:
            return None
    

    def find_customer(self):
        return self.temp_db.nearest_neighbour(self.temp_db.customers(
                self.temp_db.status_dict['n_coord'],
                exclude=[[self.temp_db.status_dict['n_items'], 0]]
            )
        )
    

    def find_depot(self):
        return self.temp_db.nearest_neighbour(self.temp_db.depots(
                self.temp_db.status_dict['n_coord'],
                # exclude=[[self.temp_db.status_dict['n_items'], 0]]
            )
        ) 

