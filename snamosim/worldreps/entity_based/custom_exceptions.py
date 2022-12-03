class EntityPlacementException(Exception):
    pass


class IntersectionError(Exception):
    def __init__(self, colliding_entities_uids, *args):
        self.colliding_entities_uids = colliding_entities_uids
        Exception(args)


