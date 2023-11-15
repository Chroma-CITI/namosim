import typing as t


class EntityPlacementException(Exception):
    pass


class IntersectionError(Exception):
    def __init__(self, colliding_entities_uids: t.List[int], *args: object):
        super().__init__(args)
        self.colliding_entities_uids = colliding_entities_uids
