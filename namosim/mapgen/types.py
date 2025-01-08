import typing as t

PERM_WALL = 2
WALL = 1
FLOOR = 0

GridCell = t.Tuple[int, int]
GridCellType = t.Literal[0, 1, 2]
