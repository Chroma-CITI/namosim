import math
import copy
import Box2D

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

import snamosim.behaviors.plan.basic_actions as ba
from snamosim.utils import utils


class CollisionPairsContactListener(Box2D.b2ContactListener):
    def __init__(self, **kwargs):
        Box2D.b2ContactListener.__init__(self, **kwargs)
        self._collision_pairs = []

    def BeginContact(self, contact):
        self._collision_pairs.append((contact.fixtureA.userData['uid'], contact.fixtureB.userData['uid']))

    def get_collision_pairs(self):
        return_value = copy.copy(self._collision_pairs)
        self._collision_pairs = []
        return return_value


class OverlappingEntitiesUidsQueryCallback(Box2D.b2QueryCallback):
    def __init__(self):
        Box2D.b2QueryCallback.__init__(self)
        self.overlapping_uids = set()

    def ReportFixture(self, fixture):
        self.overlapping_uids.add(fixture.userData['uid'])
        return True # Continue the query by returning True


class GhostData:
    def __init__(self, key, entities_polygons, entities_poses, actions, main_pose):
        self.key = key
        self.entities_polygons = entities_polygons
        self.entities_poses = entities_poses
        self.actions = actions
        self.main_pose = main_pose


class B2Sim:
    def __init__(self, entities, gravity=(0., 0.), time_step=1, velocity_iterations=1, position_iterations=1):
        """
        Initialize Box2D world using entities physics data and contact listener.
        :param entities: Dictionnary of entities with their associated uid
        :type entities: dict(int: Entity)
        :param gravity: Optional argument, in case we ever want a side-view simulation, not a top view one.
        :type gravity: tuple(float, float)
        :param time_step: Duration of a simulation time step (seconds in simulation time, not in run time)
        :type time_step: int
        :param velocity_iterations: Number of iterations to compute velocity changes on contact of dynamic obstacles
            Set to 1 by default because we don't want reactions.
        :type velocity_iterations: int
        :param position_iterations: Number of iterations to compute position changes on contact of dynamic obstacles
            Set to 1 by default because we don't want reactions.
        :type position_iterations: int
        """

        # Initialize box2d world
        self.contact_listener = CollisionPairsContactListener()
        self.b2_world = Box2D.b2World(gravity=gravity, contactListener=self.contact_listener)
        self.b2_entities = {}
        self.ghost_entities = {}

        # Initialize stepping parameters
        self.time_step = time_step
        self.velocity_iterations= velocity_iterations
        self.position_iterations= position_iterations

        # Convert entities into appropriate box2D world bodies
        self.add_entities(entities)

    def polygon_to_fixtures_defs(self, polygon, pose, uid):
        local_polygon = utils.shapely_geom_to_local(polygon, pose)
        convex_polygons_coords = utils.convert_to_convex_polygons_coordinates_list(local_polygon)
        fixtures = []
        for counter, coords in enumerate(convex_polygons_coords):
            try:
                fixture = Box2D.b2FixtureDef(
                    shape=Box2D.b2PolygonShape(vertices=coords), userData={'uid': uid}, isSensor=True
                )
                fixtures.append(fixture)
            except Exception as e:
                # Catch exceptions caused by fixture coordinates too close to make a triangle Polygon in Box2D's
                # opinion. Simply ignore the fixture.
                # TODO: Either improve convexity verification to recognize after the feast tables as convex
                #  or fix after the feast case to stop using rounded corners that are probably causing the problem
                #  when svg is read and simplified to polygon.
                pass
        return fixtures, local_polygon

    def add_entities(self, entities):
        """
        Add new entities to Box2D world
        :param entities: Dictionnary of entities with their associated uid
        :type entities: dict(int: Entity)
        """

        # Convert entities into appropriate box2D world bodies
        for uid, entity in entities.items():
            fixtures, _ = self.polygon_to_fixtures_defs(entity.polygon, entity.pose, uid)

            # Differentiate static obstacles from the rest for performance reasons
            if entity.movability == "static" or entity.movability == "unmovable":
                self.b2_entities[uid] = self.b2_world.CreateStaticBody(
                    fixtures=fixtures, position=(entity.pose[0], entity.pose[1]), angle=math.radians(entity.pose[2])
                )
            else:
                self.b2_entities[uid] = self.b2_world.CreateDynamicBody(
                    fixtures=fixtures, position=(entity.pose[0], entity.pose[1]), angle=math.radians(entity.pose[2])
                )

    def remove_entities(self, entities_uids):
        for uid in entities_uids:
            self.b2_world.DestroyBody(self.b2_entities[uid])
            del self.b2_entities[uid]

        # For the moment, just in case, we clean ghost entities just in case
        for key, body in self.ghost_entities.items():
            self.b2_world.DestroyBody(body)
        self.ghost_entities = {}

    def update_entities(self, entities, debug_init=False, debug_after=False):
        if debug_init:
            self.display_b2world()

        for uid, entity in entities.items():
            self.b2_entities[uid].position = (entity.pose[0], entity.pose[1])
            self.b2_entities[uid].angle = math.radians(entity.pose[2])

        # Have Box2D do one step to ensure aabbtree is updated
        self.b2_world.Step(
            timeStep=self.time_step,
            velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
        )

        if debug_after:
            self.display_b2world()

    def create_ghost_entity(self, key, entities_polygons, entities_poses, main_pose):
        if key in self.ghost_entities:
            return

        ghost_fixtures_defs = []
        local_poses = {}
        for uid, p in entities_polygons.items():
            fixtures, local_polygon = self.polygon_to_fixtures_defs(p, main_pose, uid)
            ghost_fixtures_defs += fixtures
            local_centroid = local_polygon.centroid.coords[0]
            local_angle = utils.angle_to_360_interval(entities_poses[uid][2] - main_pose[2])
            local_poses[uid] = (local_centroid[0], local_centroid[1], local_angle)

        self.ghost_entities[key] = self.b2_world.CreateDynamicBody(
            fixtures=ghost_fixtures_defs[0], position=(main_pose[0], main_pose[1]),
            angle=math.radians(main_pose[2]), bullet=True, userData={'local_poses': local_poses},
            active=False  # Start out as non-active (only active when used). Note: 'active' -> 'enabled' in v2.4.x
        )

    def simulate_multiple(self, ghosts_datas, stop_on_collision=True, apply=True, debug_init=True, debug_after=True):
        if debug_init:
            self.display_b2world()

        # Initialize world state
        actions_len = 0
        for ghost_data in ghosts_datas:
            # If the ghost body does not already exist, create it
            if ghost_data.key not in self.ghost_entities:
                self.create_ghost_entity(ghost_data.key, ghost_data.entities_polygons, ghost_data.entities_poses, ghost_data.main_pose)
            ghost = self.ghost_entities[ghost_data.key]

            # Activate ghost and deactivate bodies associated with the entities we made a ghost for
            ghost.active = True
            for uid in ghost_data.entities_polygons.keys():
                self.b2_entities[uid].active = False

            # Set the initial position of the ghost before executing the action sequence
            ghost.position = (ghost_data.main_pose[0], ghost_data.main_pose[1])
            ghost.angle = math.radians(ghost_data.main_pose[2])

            actions_len = max(actions_len, len(ghost_data.actions))

        collision_pairs = []
        for action_counter in range(actions_len):
            for ghost_data in ghosts_datas:
                try:
                    action = ghost_data.actions[action_counter]
                except IndexError:
                    continue

                ghost = self.ghost_entities[ghost_data.key]

                #  Try to apply the action sequence, exit as soon as we encounter the first collision
                if isinstance(action, ba.Translation):
                    ghost.linearVelocity, ghost.angularVelocity = action.compute_translation_vector(math.degrees(ghost.angle)), 0.
                elif isinstance(action, ba.Rotation):
                    ghost.linearVelocity, ghost.angularVelocity = (0., 0.), math.radians(action.angle)
                else:
                    raise TypeError(
                        "b2_collision.py simulate_multiple method only simulates Rotations and Translations."
                    )

            # Have Box2D simulate the action
            self.b2_world.Step(
                timeStep=self.time_step,
                velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
            )

            collision_pairs += self.contact_listener.get_collision_pairs()

            if stop_on_collision and collision_pairs:
                break

        # Deactivate ghost and reactivate the original entities
        for ghost_data in ghosts_datas:
            ghost = self.ghost_entities[ghost_data.key]
            ghost.linearVelocity, ghost.angularVelocity = (0., 0.), 0.
            ghost.active = False
            for uid in ghost_data.entities_polygons.keys():
                self.b2_entities[uid].active = True

        if apply:
            collision_uids = set()
            for uid_1, uid_2 in collision_pairs:
                collision_uids.add(uid_1)
                collision_uids.add(uid_2)
            for ghost_data in ghosts_datas:
                if not set(ghost_data.entities_polygons.keys()).intersection(collision_uids):
                    # If the entities associated with the ghost do not collide, apply action result to actual body
                    ghost = self.ghost_entities[ghost_data.key]
                    ghost_pose = (ghost.position[0], ghost.position[1], math.degrees(ghost.angle))
                    for uid, local_pose in ghost.userData["local_poses"].items():
                        global_centroid = utils.shapely_geom_to_global(Point(local_pose[0], local_pose[1]), ghost_pose)
                        self.b2_entities[uid].position = tuple(global_centroid.coords[0])
                        self.b2_entities[uid].angle = math.radians(utils.angle_to_360_interval(
                            local_pose[2] + math.degrees(ghost.angle)
                        ))

        if debug_after:
            self.display_b2world()

        return collision_pairs

    def check_actions_with_ghost(self, key, entities_polygons, entities_poses, actions, main_pose, debug_init=False, debug_after=False):
        # If the ghost body does not already exist, create it
        if key not in self.ghost_entities:
            self.create_ghost_entity(key, entities_polygons, entities_poses, main_pose)
        ghost = self.ghost_entities[key]

        # Activate ghost and deactivate bodies associated with the entities we made a ghost for
        ghost.active = True
        for uid in entities_polygons.keys():
            self.b2_entities[uid].active = False

        # Set the initial position of the ghost before executing the action sequence
        ghost.position, ghost.angle = (main_pose[0], main_pose[1]), math.radians(main_pose[2])

        #  Try to apply the action sequence, exit as soon as we encounter the first collision
        collision_pairs = []
        for action in actions:
            if isinstance(action, ba.Translation):
                ghost.linearVelocity, ghost.angularVelocity = action.compute_translation_vector(math.degrees(ghost.angle)), 0.
            if isinstance(action, ba.Rotation):
                ghost.linearVelocity, ghost.angularVelocity = (0., 0.), math.radians(action.angle)

            if debug_init:
                self.display_b2world()

            # Have Box2D simulate the action
            self.b2_world.Step(
                timeStep=self.time_step,
                velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
            )

            if debug_after:
                self.display_b2world()

            collision_pairs += self.contact_listener.get_collision_pairs()

            if collision_pairs:
                break

        # Deactivate ghost and reactivate the original entities
        ghost.active = False
        for uid in entities_polygons.keys():
            self.b2_entities[uid].active = True

        return collision_pairs

    def check_teleportation_with_ghost(self, key, entities_polygons, entities_poses, main_pose, debug_init=False, debug_after=False):
        # If the ghost body does not already exist, create it
        if key not in self.ghost_entities:
            self.create_ghost_entity(key, entities_polygons, entities_poses, main_pose)
        ghost = self.ghost_entities[key]

        # Activate ghost and deactivate bodies associated with the entities we made a ghost for
        ghost.active = True
        for uid in entities_polygons.keys():
            self.b2_entities[uid].active = False

        # Set the initial position of the ghost before executing the action sequence
        ghost.position, ghost.angle = (main_pose[0], main_pose[1]), math.radians(main_pose[2])

        #  Apply teleportation
        if debug_init:
            self.display_b2world()

        # Have Box2D simulate the action
        self.b2_world.Step(
            timeStep=self.time_step,
            velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
        )

        if debug_after:
            self.display_b2world()

        collision_pairs = self.contact_listener.get_collision_pairs()

        # Deactivate ghost and reactivate the original entities
        ghost.active = False
        for uid in entities_polygons.keys():
            self.b2_entities[uid].active = True

        return collision_pairs

    def deactivate_entities(self, uids):
        for uid in uids:
            if uid in self.b2_entities:
                self.b2_entities[uid].active = False

    def activate_entities(self, uids):
        for uid in uids:
            if uid in self.b2_entities:
                self.b2_entities[uid].active = True

    def get_entity_pose(self, uid):
        entity_body = self.b2_entities[uid]
        return entity_body.position[0], entity_body.position[1], math.degrees(entity_body.angle)

    def query_aabb_overlapping_uids(self, polygon):
        xmin, ymin, xmax, ymax = polygon.bounds
        aabb = Box2D.b2AABB(lowerBound=(xmin, ymin), upperBound=(xmax, ymax))
        query = OverlappingEntitiesUidsQueryCallback()
        self.b2_world.QueryAABB(query, aabb)
        return query.overlapping_uids

    def display_b2world(self):
        polygons_xy = [
            utils.shapely_geom_to_global(
                local_geom=Polygon(fixture.shape.vertices),
                local_cs_pose_in_global=(body.position[0], body.position[1], math.degrees(body.angle))
            ).exterior.xy
            for body in self.b2_world.bodies for fixture in body.fixtures
            if body.active
        ]
        fig, ax = plt.subplots()
        for polygon_xy in polygons_xy:
            ax.plot(*polygon_xy, color='black')
        ax.axis('equal')
        fig.show()
