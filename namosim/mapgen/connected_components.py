import typing as t

import numpy.typing as npt

from namosim.mapgen import utils
from namosim.mapgen.types import FLOOR, PERM_WALL, GridCell, GridCellType


class ConnectedComponents:
    def __init__(
        self,
        map: npt.NDArray[t.Any],
        component_cell_types: t.Optional[t.Set[GridCellType]] = None,
    ):
        if component_cell_types is None:
            self.component_cell_types: t.Set[GridCellType] = {FLOOR}
        else:
            self.component_cell_types = component_cell_types
        self.components: t.Dict[int, t.Set[GridCell]] = {}
        self.cell_to_component: t.Dict[GridCell, int] = {}
        self.map = map
        self.compute_components(map)

    def compute_components(self, map: npt.NDArray[t.Any]):
        self.components = {}
        self.cell_to_component = {}
        R, C = map.shape
        for r in range(R):
            for c in range(C):
                cell = (r, c)
                if cell in self.cell_to_component:
                    continue

                if map[r][c] in self.component_cell_types:
                    component_idx = len(self.components)
                    assert component_idx not in self.components
                    accessible_cells = self.get_accessible_cells(start=cell, map=map)
                    self.components[component_idx] = accessible_cells
                    for x in accessible_cells:
                        self.cell_to_component[x] = component_idx

    def get_smallest_component(self) -> t.Tuple[int, t.Set[GridCell]]:
        n = float("inf")
        smallest = 0
        for comp in self.components.keys():
            if len(self.components[comp]) < n:
                smallest = comp
        return smallest, self.components[smallest]

    def find_nearest_neighbor(
        self, component_idx: int
    ) -> t.Tuple[int | None, t.List[GridCell]]:
        if len(self.components) < 2:
            return None, []

        visited: t.Set[GridCell] = set()
        component = self.components[component_idx]
        frontier = list(component)
        came_from: t.Dict[GridCell, GridCell] = {}

        def get_path(end: GridCell, d: t.Dict[GridCell, GridCell]) -> t.List[GridCell]:
            result = [end]
            while end in d:
                result.append(d[end])
                end = d[end]
            result.reverse()
            return result

        while len(frontier) > 0:
            x = frontier.pop()

            if (
                x in self.cell_to_component
                and self.cell_to_component[x] != component_idx
            ):
                return self.cell_to_component[x], get_path(x, came_from)

            visited.add(x)
            for n in self.get_neighbors(x, self.map, include_walls=True):
                if n in visited or n in component:
                    continue
                frontier.append(n)
                came_from[n] = x

        return None, []

    def get_accessible_cells(
        self, start: GridCell, map: npt.NDArray[t.Any]
    ) -> t.Set[GridCell]:
        frontier = [start]
        visited: t.Set[GridCell] = set()
        while len(frontier):
            x = frontier.pop()
            visited.add(x)

            for n in self.get_neighbors(x, map):
                if n not in visited:
                    frontier.append(n)

        return visited

    def get_neighbors(
        self, cell: GridCell, map: npt.NDArray[t.Any], include_walls: bool = False
    ) -> t.Iterable[GridCell]:
        (r, c) = cell
        candidates = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        for cand in candidates:
            if not utils.is_in_map(cand[0], cand[1], map):
                continue

            if include_walls and self.map[cand[0]][cand[1]] != PERM_WALL:
                yield cand
            elif utils.is_empty(cand[0], cand[1], map, self.component_cell_types):
                yield cand
