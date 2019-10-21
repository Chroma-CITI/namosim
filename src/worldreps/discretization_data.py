# TODO: Extract self.inflation_radius = inflation_radius
#         self.cost_lethal = cost_lethal
#         self.cost_inscribed = cost_inscribed
#         self.cost_circumscribed = cost_circumscribed
#         self.cost_possibly_nonfree = cost_possibly_nonfree
#  from the class into a separate inflation_radius field and ProbabilisticInflationData class


class DiscretizationData:
    def __init__(self, res, inflation_radius, cost_lethal, cost_inscribed, cost_circumscribed, cost_possibly_nonfree):
        self.res = res
        self.inflation_radius = inflation_radius
        self.cost_lethal = cost_lethal
        self.cost_inscribed = cost_inscribed
        self.cost_circumscribed = cost_circumscribed
        self.cost_possibly_nonfree = cost_possibly_nonfree
        self.grid_pose = (0.0, 0.0, 0.0)
        self.width = 0.0
        self.height = 0.0
        self.d_width = 0
        self.d_height = 0

        self.saved_hash = self.__hash__()

    def __key(self):
        return (self.res, self.inflation_radius,
                self.cost_lethal, self.cost_inscribed, self.cost_circumscribed, self.cost_possibly_nonfree,
                self.grid_pose, self.width, self.height, self.d_width, self.d_height)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, DiscretizationData):
            return self.__key() == other.__key()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
