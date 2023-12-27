import os

import utils

from namosim.world.entity import Movability
from namosim.world.obstacle import Obstacle
from namosim.world.world import World


def concave_to_convex_obstacles(world_json_path):
    world = World.load_from_json(world_json_path)
    entities_uids = list(world.entities.keys())
    for uid in entities_uids:
        entity = world.entities[uid]
        convex_polygons = utils.convert_to_convex_polygons_list(entity.polygon)
        if (
            entity.polygon is not convex_polygons[0]
            and isinstance(entity, Obstacle)
            and entity.movability == Movability.STATIC
        ):
            for counter, polygon in enumerate(convex_polygons):
                center = polygon.centroid.coords[0]
                new_entity = Obstacle(
                    name=entity.name + "_" + str(counter),
                    polygon=polygon,
                    pose=(center[0], center[1], 0.0),
                    full_geometry_acquired=entity.full_geometry_acquired,
                    type_=entity.type_,
                    movability=entity.movability,
                    style=entity.style,
                )
                world.add_entity(new_entity)
            world.remove_entity(uid)

    new_world_json_path = os.path.splitext(world_json_path)[0] + "_new.json"
    world.save_to_files(json_filepath=new_world_json_path)


if __name__ == "__main__":
    concave_to_convex_obstacles(
        os.path.join(
            __file__,
            "../../../data/simulations/iros_2021/citi/1_robots/100_goals/0007/world_0007.json",
        )
    )
