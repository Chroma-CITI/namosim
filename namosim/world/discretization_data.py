from namosim.data_models import PoseModel


class DiscretizationData:
    """
    This class contains information about how the world is discretized or "rasterized" into a rectangular grid of cells.
    """

    def __init__(
        self,
        res: float,
        grid_pose: PoseModel = (0.0, 0.0, 0.0),
        width: float = 0.0,
        height: float = 0.0,
        d_width: int = 0,
        d_height: int = 0,
    ):
        self.res = res
        self.grid_pose = grid_pose
        self.width = width
        self.height = height
        self.d_width = d_width
        self.d_height = d_height

        self.saved_hash = self.__hash__()

    def __key(self):
        return (
            self.res,
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
