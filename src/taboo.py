from entity import Entity


class Taboo(Entity):

    def __init__(self, name, polygon, dd, cost, uid=0):
        Entity.__init__(self, name, polygon, dd, uid)
        self.cost = cost
