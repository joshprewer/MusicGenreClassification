from pyharmonysearch import ObjectiveFunctionInterface, harmony_search

class ObjectiveFunction(ObjectiveFunctionInterface):

    def __init__(self):
    self._lower_bounds = [0]
    self._upper_bounds = [1]
    self._variable = [True]

    # define all input parameters
    self._maximize = True  # do we maximize or minimize?
    self._max_imp = 50000  # maximum number of improvisations
    self._hms = 50  # harmony memory size
    self._hmcr = 0.99  # harmony memory considering rate
    self._par = 0.5  # pitch adjusting rate
    self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
    self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        return False

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return False

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize