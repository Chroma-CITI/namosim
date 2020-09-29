from src.worldreps.entity_based.obstacle import Obstacle

import numpy as np
import copy


class OmniscientSensor:
    def __init__(self):
        self.parent_uid = None

    def update_from_fov(self, reference_world, target_world):
        robot_pose = reference_world.entities[self.parent_uid].pose
        for entity_uid, reference_entity in reference_world.entities.items():
            if entity_uid != self.parent_uid:
                if entity_uid in target_world.entities:
                    target_entity = target_world.entities[entity_uid]
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
                else:
                    target_world.add_entity(reference_entity.light_copy())

    def translate(self, xoff, yoff):
        pass

    def rotate(self,angle, rot_center='centroid'):
        pass

    def to_json(self):
        return {"type": "omniscient"}