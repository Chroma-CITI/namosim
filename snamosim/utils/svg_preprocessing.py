import os
import utils
from snamosim.worldreps.entity_based.world import World
from snamosim.worldreps.entity_based.obstacle import Obstacle


def concave_to_convex_obstacles(world_json_path):
    world = World.load_from_json(world_json_path)
    entities_uids = list(world.entities.keys())
    for uid in entities_uids:
        entity = world.entities[uid]
        convex_polygons = utils.convert_to_convex_polygons_list(entity.polygon)
        if not(entity.polygon is convex_polygons[0]) and isinstance(entity, Obstacle) and entity.movability == "static":
            for counter, polygon in enumerate(convex_polygons):
                center = polygon.centroid.coords[0]
                new_entity = Obstacle(
                    name=entity.name + "_" + str(counter), polygon=polygon, pose=(center[0], center[1], 0.),
                    full_geometry_acquired=entity.full_geometry_acquired, type_in=entity.type,
                    movability=entity.movability
                )
                world.add_entity(new_entity)
            world.remove_entity(uid)

    new_world_json_path = os.path.splitext(world_json_path)[0] + "_new.json"
    world.save_to_files(json_filepath=new_world_json_path)


if __name__ == '__main__':
    concave_to_convex_obstacles("/home/xia0ben/INRIA/Code/s-namo-sim/data/simulations/iros_2021/citi/1_robots/100_goals/0007/world_0007.json")
