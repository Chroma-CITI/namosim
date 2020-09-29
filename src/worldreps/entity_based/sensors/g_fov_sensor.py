from src.worldreps.entity_based.obstacle import Obstacle
from src.worldreps.entity_based.sensors.circular_sector_sensor import CircularSectorSensor

import numpy as np

from shapely.ops import cascaded_union
from shapely.errors import TopologicalError
from shapely.geometry import Polygon


class GFOVSensor(CircularSectorSensor):
    def __init__(self, fov_max_radius, fov_min_radius, fov_opening_angle, parent_entity_pose):
        CircularSectorSensor.__init__(self, fov_max_radius, fov_min_radius, fov_opening_angle, parent_entity_pose)

    def _get_entities_in_fov_seethrough(self, world):
        entities_in_fov = dict()

        for entity_uid, entity in world.entities.items():
            if entity_uid != self.parent_uid and entity.polygon.intersects(self.fov_polygon):
                if entity.polygon.within(self.fov_polygon):
                    entity_visible_polygon = entity.polygon
                    full_geometry_acquired = True
                else:
                    try:
                        entity_visible_polygon = entity.polygon.difference(
                            entity.polygon.difference(self.fov_polygon))
                        full_geometry_acquired = False
                    except TopologicalError:
                        continue  # If we could not make a polygon, do not try to create Entity

                if isinstance(entity_visible_polygon, Polygon):
                    entities_in_fov[entity_uid] = Obstacle(name="unknown",
                                                           polygon=entity_visible_polygon,
                                                           pose=[entity_visible_polygon.centroid.coords[0][0],
                                                                 entity_visible_polygon.centroid.coords[0][1],
                                                                 entity.pose[2]],
                                                           full_geometry_acquired=full_geometry_acquired,
                                                           type_in="unknown",
                                                           uid=entity_uid)
        return entities_in_fov

    def update_from_fov(self, reference_world, target_world):
        reference_entities = self._get_entities_in_fov_seethrough(reference_world)

        for entity_uid, reference_entity in reference_entities.items():
            if isinstance(reference_entity, Obstacle) and entity_uid != self.parent_uid:
                # If entity is already registered, update it
                try:
                    target_entity = target_world.entities[entity_uid]
                    # If self entity full geometry has not been acquired, update it
                    if not target_entity.full_geometry_acquired:
                        if reference_entity.full_geometry_acquired:
                            target_world.set_entity_polygon(target_entity.uid, reference_entity.polygon, True)
                        else:
                            partial_polygon = cascaded_union(
                                [target_entity.polygon, reference_entity.polygon]).convex_hull
                            target_world.set_entity_polygon(target_entity.uid, partial_polygon)
                    # If it is already known, only translate/rotate the polygon appropriately
                    else:
                        if reference_entity.full_geometry_acquired:
                            translation = [reference_entity.pose[0] - target_entity.pose[0],
                                           reference_entity.pose[1] - target_entity.pose[1]]
                            rotation = (reference_entity.pose[2] - target_entity.pose[2]) % 360.
                            # Only apply translation if there is one
                            if not all(np.isclose(translation, [0., 0.], rtol=0.00001)):
                                target_world.translate_entity(entity_uid, translation)
                            # Only apply rotation if there is one
                            if rotation != 0:
                                target_world.rotate_entity(entity_uid, rotation)
                # If entity is not registered yet, create it
                except KeyError:
                    target_world.add_entity(reference_entity)

    def to_json(self):
        return {
            "type": "perfect_g_fov",
            "min_radius": self.fov_min_radius,
            "max_radius": self.fov_max_radius,
            "opening_angle": self.fov_opening_angle
        }
