class Taboo:
    last_id = 1

    def __init__(self, name, polygon, uid=0):
        if uid == 0:
            self.uid = Taboo.last_id
            Taboo.last_id = Taboo.last_id + 1
        else:
            self.uid = uid

        self.name = name
        self.polygon = polygon
