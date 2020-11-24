from snamosim.worldreps.entity_based.obstacle import Obstacle
from snamosim.worldreps.entity_based.sensors.circular_sector_sensor import CircularSectorSensor


class SFOVSensor(CircularSectorSensor):
    def __init__(self, fov_max_radius, fov_min_radius, fov_opening_angle, parent_entity_pose):
        CircularSectorSensor.__init__(self, fov_max_radius, fov_min_radius, fov_opening_angle, parent_entity_pose)

    def _get_entities_in_fov_seethrough(self, world):
        entities_in_fov = dict()

        for entity_uid, entity in world.entities.items():
            if entity_uid != self.parent_uid and entity.polygon.within(self.fov_polygon):
                entities_in_fov[entity_uid] = Obstacle(name=entity.name,
                                                       polygon=entity.polygon,
                                                       pose=entity.pose,
                                                       full_geometry_acquired=True,
                                                       type_in=entity.type,
                                                       uid=entity_uid)
        return entities_in_fov

    def update_from_fov(self, reference_world, target_world):
        reference_entities = self._get_entities_in_fov_seethrough(reference_world)

        for entity_uid, reference_entity in reference_entities.items():
            if isinstance(reference_entity, Obstacle) and entity_uid != self.parent_uid:
                # If entity is already registered, update it
                try:
                    target_world.entities[entity_uid].name = reference_entity.name
                    target_world.entities[entity_uid].type = reference_entity.type

                # If entity is not registered yet, create it
                except KeyError as e:
                    raise (e, "Update with S FoV sensor should never need to create an object !")

    def to_json(self):
        return {
            "type": "perfect_s_fov",
            "min_radius": self.fov_min_radius,
            "max_radius": self.fov_max_radius,
            "opening_angle": self.fov_opening_angle
        }
