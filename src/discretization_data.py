class DiscretizationData:
    def __init__(self, res, inflation_radius, cost_lethal, cost_inscribed, cost_circumscribed, cost_possibly_nonfree):
        self.res = res
        self.inflation_radius = inflation_radius
        self.cost_lethal = cost_lethal
        self.cost_inscribed = cost_inscribed
        self.cost_circumscribed = cost_circumscribed
        self.cost_possibly_nonfree = cost_possibly_nonfree
        self.grid_pose = [0.0, 0.0, 0.0]
