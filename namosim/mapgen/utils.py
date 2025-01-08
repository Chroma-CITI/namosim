import typing as t

import numpy.typing as npt

from namosim.mapgen.types import GridCellType


def is_in_map(r: int, c: int, map: npt.NDArray[t.Any]) -> bool:
    R, C = map.shape
    return not (r < 0 or r >= R or c < 0 or c >= C)


def is_empty(
    r: int, c: int, map: npt.NDArray[t.Any], empty_cell_types: t.Set[GridCellType]
) -> bool:
    if is_in_map(r, c, map):
        return map[r][c] in empty_cell_types
    return False
