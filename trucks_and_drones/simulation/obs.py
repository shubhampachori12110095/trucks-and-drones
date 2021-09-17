import numpy as np
from gym import spaces

def normalize_safe(vals, lows, highs):
    for i in range(len(vals)):
        if highs[i] is None or (highs[i] - lows[i]) == 0:
            vals[i] = np.tanh(vals[i]*0.1)
        else:
            vals[i] = (vals[i] - lows[i]) / (highs[i] - lows[i])
    return vals

class DiscreteObserver:

    def __init__(
            self,
            database,
            keys: (str, list),
            bins: int = None,
    ):

        self.temp_db = database
        self.keys = keys

        self.num_obs = self.temp_db.count_elements(self.keys)
        self.lows = self.temp_db.minimas(self.keys)
        self.highs = self.temp_db.maximas(self.keys)

        if bins is None:
            self.bins = np.nanmax(self.highs-self.lows)
        else:
            self.bins = bins

        if self.bins == np.nan or self.bins is None:
            raise Exception('Failed to calculate number of bins for keys:'+str(keys))

    def __call__(self):

        vals = np.ravel(self.temp_db(self.keys))
        vals = normalize_safe(vals, self.lows, self.highs)

        if isinstance(vals, str):
            return self._get_discrete(vals)

        discrete_array = np.array([])
        for val in vals:
            np.append(discrete_array, self._get_discrete(val))
        return discrete_array

    def _get_discrete(self, val):
        val = int(val * self.bins)
        placeholder_array = np.zeros((self.bins))
        placeholder_array[val] = 1

    @property
    def gym_space(self):
        return spaces.Discrete(self.bins*self.num_obs)



class ContinObserver:

    def __init__(
            self,
            database,
            keys: (str, list),
            normalize: bool = False,
    ):

        self.temp_db = database
        self.keys = keys
        self.normalize = normalize

        self.num_obs = self.temp_db.count_elements(self.keys)
        self.lows = self.temp_db.minimas(self.keys)
        self.highs = self.temp_db.maximas(self.keys)

    def __call__(self):
        vals = np.ravel(self.temp_db(self.keys))
        if self.normalize:
            vals = normalize_safe(vals, self.lows, self.highs)
        return vals

    @property
    def gym_space(self):
        if self.normalize:
            return spaces.Box(low=np.zeros((self.num_obs)), high=np.ones((self.num_obs)),  dtype=np.float32)
        return spaces.Box(low=self.lows, high=self.highs,  dtype=np.float32)
