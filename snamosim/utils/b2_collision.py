import math
import copy
import Box2D

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

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


class B2Sim:
    def __init__(self, entities, gravity=(0., 0.)):
        """
        Initialize Box2D world using entities physics data and contact listener.
        :param entities: Dictionnary of entities with their associated uid
        :type entities: dict(int: Entity)
        :param gravity: Optional argument, in case we ever want a side-view simulation, not a top view one.
        :type gravity: tuple(float, float)
        """

        # Initialize box2d world
        self.contact_listener = CollisionPairsContactListener()
        self.b2_world = Box2D.b2World(gravity=gravity, contactListener=self.contact_listener)
        self.b2_entities = {}
        self.ghost_entities = {}

        # Convert entities into appropriate box2D world bodies
        self.add_entities(entities)

    def add_entities(self, entities):
        """
        Add new entities to Box2D world
        :param entities: Dictionnary of entities with their associated uid
        :type entities: dict(int: Entity)
        """

        # Convert entities into appropriate box2D world bodies
        for uid, entity in entities.items():
            local_polygon = utils.shapely_geom_to_local(entity.polygon, entity.pose)
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
        for key, body in self.ghost_entities:
            self.b2_world.DestroyBody(body)
        self.b2_entities = {}

    def update_entities(self, entities):
        for uid, entity in entities.items():
            self.b2_entities[uid].position = (entity.pose[0], entity.pose[1])
            self.b2_entities[uid].angle = math.radians(entity.pose[2])

    def create_ghost_entity(self, key, entities_polygons, main_pose):
        if key in self.ghost_entities:
            return

        ghost_fixtures_defs = []
        local_centroids = {}
        for uid, p in entities_polygons.items():
            local_polygon = utils.shapely_geom_to_local(p, main_pose)
            ghost_fixtures_defs.append(Box2D.b2FixtureDef(shape=local_polygon, userData={'uid': uid}, isSensor=True))
            self.b2_entities[uid].active = False

        self.ghost_entities[key] = self.b2_world.CreateDynamicBody(
            fixtures=ghost_fixtures_defs[0], position=(main_pose[0], main_pose[1]),
            angle=math.radians(main_pose[2]), bullet=True, userData={'local_centroids': local_centroids},
            active=False  # Start out as non-active (only active when used). Note: 'active' -> 'enabled' in v2.4.x
        )

    def check_actions_with_ghost(self, key, entities_polygons, actions, main_pose, debug_init=False, debug_after=False):
        # If the ghost body does not already exist, create it
        if key not in self.ghost_entities:
            self.create_ghost_entity(key, entities_polygons, main_pose)
        ghost = self.ghost_entities[key]

        # Activate ghost and deactivate bodies associated with the entities we made a ghost for
        ghost.active = True
        for uid in entities_polygons.keys():
            self.b2_entities[uid].active = False

        # Set the initial position of the ghost before executing the action sequence
        ghost.position, ghost.angle = (main_pose[0], main_pose[1]), math.radians(main_pose[2])

        #  Try to apply the action sequence, exit as soon as we encounter the first collision
        for action in actions:
            if isinstance(action, ba.Translation):
                ghost.linearVelocity, ghost.angularVelocity = action.compute_translation_vector(main_pose), 0.
            if isinstance(action, ba.Rotation):
                ghost.linearVelocity, ghost.angularVelocity = (0., 0.), math.radians(action.angle)

            if debug_init:
                self.display_b2world()

            # Have Box2D simulate the action
            self.b2_world.Step(timeStep=1, velocityIterations=1, positionIterations=1)

            if debug_after:
                self.display_b2world()

            collision_pairs = self.contact_listener.get_collision_pairs()

            if collision_pairs:
                return collision_pairs

        # Deactivate ghost and reactivate the original entities
        ghost.active = False
        for uid in entities_polygons.keys():
            self.b2_entities[uid].active = True

        return []

    def check_teleportation_with_ghost(self, key, entities_polygons, main_pose, debug_init=False, debug_after=False):
        # If the ghost body does not already exist, create it
        if key not in self.ghost_entities:
            self.create_ghost_entity(key, entities_polygons, main_pose)
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
        self.b2_world.Step(timeStep=1, velocityIterations=1, positionIterations=1)

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
