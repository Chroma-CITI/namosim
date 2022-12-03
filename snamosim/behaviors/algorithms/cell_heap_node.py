class CellHeapNode:
    def __init__(self, cost, cell):
        self.cost = cost
        self.cell = cell

    def __cmp__(self, other):
        return cmp(self.cost, other.cost)

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost and self.cell == other.cell