# TODO: Extract self.inflation_radius = inflation_radius
#         self.cost_lethal = cost_lethal
#         self.cost_inscribed = cost_inscribed
#         self.cost_circumscribed = cost_circumscribed
#         self.cost_possibly_nonfree = cost_possibly_nonfree
#  from the class into a separate inflation_radius field and ProbabilisticInflationData class


from namosim.models import PoseModel


class DiscretizationData:
    def __init__(
        self,
        res: float,
        inflation_radius: float,
        cost_lethal: float = 1.0,
        cost_inscribed: float = 0.5,
        cost_circumscribed: float = 0.25,
        cost_possibly_nonfree: float = 0.10,
        grid_pose: PoseModel = (0.0, 0.0, 0.0),
        width: float = 0.0,
        height: float = 0.0,
        d_width: int = 0,
        d_height: int = 0,
    ):
        self.res = res
        self.inflation_radius = inflation_radius
        self.cost_lethal = cost_lethal
        self.cost_inscribed = cost_inscribed
        self.cost_circumscribed = cost_circumscribed
        self.cost_possibly_nonfree = cost_possibly_nonfree
        self.grid_pose = grid_pose
        self.width = width
        self.height = height
        self.d_width = d_width
        self.d_height = d_height

        self.saved_hash = self.__hash__()

    def __key(self):
        return (
            self.res,
            self.inflation_radius,
            self.cost_lethal,
            self.cost_inscribed,
            self.cost_circumscribed,
            self.cost_possibly_nonfree,
            self.grid_pose,
            self.width,
            self.height,
            self.d_width,
            self.d_height,
        )

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other: object):
        if isinstance(other, DiscretizationData):
            return self.__key() == other.__key()
        return False

    def __ne__(self, other: object):
        return not self.__eq__(other)
