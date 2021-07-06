import math
import numpy
import os
import Box2D

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

import snamosim.behaviors.plan.basic_actions as ba
from snamosim.utils import utils


class CollisionPairsContactListener(Box2D.b2ContactListener):
    def __init__(self, **kwargs):
        Box2D.b2ContactListener.__init__(self, **kwargs)
        self._collision_pairs = set()

    def BeginContact(self, contact):
        self._collision_pairs.add((contact.fixtureA.userData['uid'], contact.fixtureB.userData['uid']))

    def get_collision_pairs(self, ignored_collision_pairs=None):
        if ignored_collision_pairs:
            collision_pairs = self._collision_pairs.difference(ignored_collision_pairs)
        else:
            collision_pairs = self._collision_pairs
        self._collision_pairs = set()
        return collision_pairs

    def reset(self):
        self._collision_pairs = set()


class OverlappingEntitiesUidsQueryCallback(Box2D.b2QueryCallback):
    def __init__(self):
        Box2D.b2QueryCallback.__init__(self)
        self.overlapping_uids = set()

    def ReportFixture(self, fixture):
        self.overlapping_uids.add(fixture.userData['uid'])
        return True # Continue the query by returning True


class B2Sim:
    def __init__(self, entities, gravity=(0., 0.), frequency=5, velocity_iterations=1, position_iterations=1):
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
        self.b2_joints = {}
        self.deactivated_entities = {}

        # Initialize stepping parameters
        self.time_step = 1./float(frequency)
        self.frequency = frequency
        self.velocity_iterations= velocity_iterations
        self.position_iterations= position_iterations

        # Convert entities into appropriate box2D world bodies
        self.add_entities(entities)

    @staticmethod
    def polygon_to_fixtures_defs(polygon, pose, uid):
        local_polygon = utils.shapely_geom_to_local(polygon, pose)
        convex_polygons_coords = utils.convert_to_convex_polygons_coordinates_list(local_polygon)
        fixtures_defs = []
        for counter, coords in enumerate(convex_polygons_coords):
            try:
                fixture_def = Box2D.b2FixtureDef(
                    friction=0.,
                    restitution=0.,
                    density=1.,
                    isSensor=True,
                    categoryBits=1,
                    maskBits=65535,
                    groupIndex=0,
                    shape=Box2D.b2PolygonShape(vertices=coords),
                    userData={'uid': uid}
                )
                Box2D.b2FixtureDef(
                    shape=Box2D.b2PolygonShape(vertices=coords),
                )
                fixtures_defs.append(fixture_def)
            except Exception as e:
                # Catch exceptions caused by fixture coordinates too close to make a triangle Polygon in Box2D's
                # opinion. Simply ignore the fixture.
                # TODO: Either improve convexity verification to recognize after the feast tables as convex
                #  or fix after the feast case to stop using rounded corners that are probably causing the problem
                #  when svg is read and simplified to polygon.
                pass
        return fixtures_defs

    def add_entities(self, entities):
        """
        Add new entities to Box2D world
        :param entities: Dictionnary of entities with their associated uid
        :type entities: dict(int: Entity)
        """
        for uid, entity in entities.items():
            self.b2_entities[uid] = self.b2_world.CreateBody(
                defn=Box2D.b2BodyDef(
                    type=0 if entity.movability == "unmovable" or entity.movability == "static" else 2,
                    position=(entity.pose[0], entity.pose[1]),
                    angle=math.radians(entity.pose[2]),
                    linearVelocity=(0., 0.),
                    angularVelocity=0.,
                    linearDamping=0.,
                    angularDamping=0.,
                    allowSleep=True,
                    awake=True,
                    fixedRotation=False,
                    bullet=False if entity.movability == "unmovable" or entity.movability == "static" else True,
                    active=True,
                    gravityScale=1.,
                    userData={'uid': uid}
                )
            )
            for fixture_def in self.polygon_to_fixtures_defs(entity.polygon, entity.pose, uid):
                try:
                    self.b2_entities[uid].CreateFixture(defn=fixture_def)
                except AssertionError:
                    continue

    def remove_entities(self, entities_uids):
        for uid in entities_uids:
            self.b2_world.DestroyBody(self.b2_entities[uid])
            del self.b2_entities[uid]

    def update_entities(self, entities, debug_init=False, debug_after=False):
        if debug_init:
            self.display_b2world()

        for uid, entity in entities.items():
            b2_entity = self.b2_entities[uid]
            b2_entity.position = (entity.pose[0], entity.pose[1])
            b2_entity.angle = math.radians(entity.pose[2])

        # Have Box2D do one step to ensure aabbtree is updated
        for i in range(self.frequency):
            self.b2_world.Step(
                timeStep=self.time_step,
                velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
            )

        if debug_after:
            self.display_b2world()

    def get_collides_with(self, ignored_collision_pairs):
        collision_pairs = self.contact_listener.get_collision_pairs(ignored_collision_pairs)
        if ignored_collision_pairs:
            collision_pairs = collision_pairs.difference(ignored_collision_pairs)
        else:
            collision_pairs = collision_pairs
        collides_with = {}
        for uid_1, uid_2 in collision_pairs:
            if uid_1 in collides_with:
                collides_with[uid_1].add(uid_2)
            else:
                collides_with[uid_1] = {uid_2}
            if uid_2 in collides_with:
                collides_with[uid_2].add(uid_1)
            else:
                collides_with[uid_2] = {uid_1}
        return collides_with

    def simulate_simple_kinematics(self, list_of_uid_to_actions, init_poses=None, init_welded_pairs=None,
                                   apply=False, return_poses_before_reset=False, debug_all=False, debug_before_init=False, debug_after_init=False,
                                   debug_after_reset=False, debug_after_each_step=False):
        """
        Initiliazes bodies poses and joints in Box2D simulation at the specified state, then try to apply the list
        of actions for each entity. Returns the dictionnary of colliding entities as soon as at least one collision
        is detected during a single time step. If no actions are given, this function can be used to check if the
        specified initial state is in collision or not. Unless the "apply" parameter is True, the simulation state
        is reset as it was before calling this function.
        :param list_of_uid_to_actions:
        :type list_of_uid_to_actions:
        :param init_poses:
        :type init_poses:
        :param init_welded_pairs:
        :type init_welded_pairs:
        :param apply:
        :type apply:
        :param debug_all:
        :type debug_all:
        :param debug_before_init:
        :type debug_before_init:
        :param debug_after_init:
        :type debug_after_init:
        :param debug_after_reset:
        :type debug_after_reset:
        :param debug_after_each_step:
        :type debug_after_each_step:
        :return:
        :rtype:
        """
        # Initialize world state
        # TODO Make it so we can choose to break at first obstacle that collides or accumulate the ids of those that do

        init_poses = init_poses if init_poses is not None else dict()
        init_welded_pairs = init_welded_pairs if init_welded_pairs is not None else set()

        previous_poses = self._initialize_world_state(
            init_poses, init_welded_pairs, debug=debug_all or debug_before_init
        )

        collides_with = self.get_collides_with(init_welded_pairs)

        if debug_all or debug_after_init:
             self.display_b2world(name="After initial world state is set")

        if collides_with:
            poses_before_reset = {
                uid: (
                    self.b2_entities[uid].position[0],
                    self.b2_entities[uid].position[1],
                    math.degrees(self.b2_entities[uid].angle)
                )
                for uid in previous_poses.keys()
            }
            if not apply:
                self._reset_world_state(previous_poses, init_welded_pairs, [], debug_all or debug_after_reset)
            else:
                # If apply, at least reset the colliding entities
                colliding_uids = set(collides_with.keys())
                self._reset_world_state(
                    {uid: pose for uid, pose in previous_poses if uid in colliding_uids},
                    {pair for pair in init_welded_pairs if pair[0] in colliding_uids or pair[1] in colliding_uids},
                    [], debug_all or debug_after_reset
                )
            if return_poses_before_reset:
                return collides_with, poses_before_reset
            else:
                return collides_with

        # Try to apply actions
        nb_actions = len(list_of_uid_to_actions)
        welded_pairs, unwelded_pairs = set(), set()
        for action_index, uid_to_action in enumerate(list_of_uid_to_actions):
            ignored_collision_pairs = set()
            for uid, action in uid_to_action.items():
                b2_entity = self.b2_entities[uid]
                if isinstance(action, ba.Translation):
                    if isinstance(action, (ba.Grab, ba.Release)):
                        pair, reverse_pair = (uid, action.entity_uid), (action.entity_uid, uid)
                        ignored_collision_pairs.add(pair)
                        ignored_collision_pairs.add(reverse_pair)
                        if isinstance(action, ba.Release):
                            if pair in self.b2_joints:
                                self._unweld(pair)
                                unwelded_pairs.add(pair)
                    b2_entity.linearVelocity, b2_entity.angularVelocity = action.compute_translation_vector(
                        math.degrees(b2_entity.angle)), 0.
                elif isinstance(action, ba.Rotation):
                    b2_entity.linearVelocity, b2_entity.angularVelocity = (0., 0.), math.radians(action.angle)
                else:
                    raise TypeError(
                        "b2_collision.py simulate_multiple method only simulates Rotations and Translations."
                    )

            # Have Box2D simulate the action
            for i in range(self.frequency):
                self.b2_world.Step(
                    timeStep=self.time_step,
                    velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
                )
                if debug_all or debug_after_each_step:
                    self.display_b2world(name="After applying substep {}/{} instep {}/{}".format(i + 1, self.frequency, action_index + 1, nb_actions))
                    pass

            # Reset velocities and apply weld joint for grab actions
            for uid, action in uid_to_action.items():
                b2_entity = self.b2_entities[uid]
                b2_entity.linearVelocity, b2_entity.angularVelocity = (0., 0.), 0.
                if isinstance(action, ba.Grab):
                    pair = (uid, action.entity_uid)
                    if pair not in self.b2_joints:
                        self._weld(pair)
                        welded_pairs.add(pair)

            collides_with = self.get_collides_with(init_welded_pairs)

            if debug_all or debug_after_each_step:
                self.display_b2world(name="After applying step {}/{}".format(action_index + 1, nb_actions))

            if collides_with:
                if debug_all or debug_after_each_step:
                    self.display_b2world(name="Collision after applying step {}/{}".format(action_index + 1, nb_actions), show=True)

                poses_before_reset = {
                    uid: (
                        self.b2_entities[uid].position[0],
                        self.b2_entities[uid].position[1],
                        math.degrees(self.b2_entities[uid].angle)
                    )
                    for uid in previous_poses.keys()
                }
                if not apply:
                    self._reset_world_state(previous_poses, init_welded_pairs.union(welded_pairs), unwelded_pairs.difference(init_welded_pairs), debug_all or debug_after_reset)
                else:
                    # If apply, at least reset the colliding entities
                    colliding_uids = set(collides_with.keys())
                    self._reset_world_state(
                        {uid: pose for uid, pose in previous_poses if uid in colliding_uids},
                        {pair for pair in init_welded_pairs.union(welded_pairs) if pair[0] in colliding_uids or pair[1] in colliding_uids},
                        {pair for pair in unwelded_pairs.difference(init_welded_pairs) if pair[0] in colliding_uids or pair[1] in colliding_uids},
                        debug_all or debug_after_reset
                    )
                if return_poses_before_reset:
                    return collides_with, poses_before_reset
                else:
                    return collides_with

        # Reset world state
        poses_before_reset = {
            uid: (
                self.b2_entities[uid].position[0],
                self.b2_entities[uid].position[1],
                math.degrees(self.b2_entities[uid].angle)
            )
            for uid in previous_poses.keys()
        }
        if not apply:
            self._reset_world_state(previous_poses, init_welded_pairs.union(welded_pairs), unwelded_pairs.difference(init_welded_pairs), debug_all or debug_after_reset)

        if return_poses_before_reset:
            return collides_with, poses_before_reset
        else:
            return collides_with

    def _initialize_world_state(self, init_poses, init_welded_pairs, debug=False):
        if debug:
            self.display_b2world(name="Before initial world state is set")

        previous_poses = {}  # CAREFUL, ANGLES ARE SAVED IN RADIANS !
        for uid, pose in init_poses.items():
            b2_entity = self.b2_entities[uid]
            previous_poses[uid] = (b2_entity.position[0], b2_entity.position[1], b2_entity.angle)
            b2_entity.position, b2_entity.angle = (pose[0], pose[1]), math.radians(pose[2])
            b2_entity.angularVelocity = -0.01

        for pair in init_welded_pairs:
            self._weld(pair)

        # Have Box2D apply the initial state, ignore collisions to work around Weld Joint teleport phenomenon...
        for i in range(self.frequency):
            self.b2_world.Step(
                timeStep=self.time_step,
                velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
            )
            if debug:
                self.display_b2world(name="Before initial world state is set - Init Steps")
        self.contact_listener.reset()

        # Slightly oscillate back 0.01 rad to force proper collision checks updates
        for uid in init_poses.keys():
            self.b2_entities[uid].angularVelocity = 0.01

        for i in range(self.frequency):
            self.b2_world.Step(
                timeStep=self.time_step,
                velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
            )
            if debug:
                self.display_b2world(name="Before initial world state is set - Init Steps")

        return previous_poses

    def _reset_world_state(self, previous_poses, welded_pairs, unwelded_pairs, debug=False):
        for uid, prev_pose in previous_poses.items():
            b2_entity = self.b2_entities[uid]
            b2_entity.position, b2_entity.angle = (prev_pose[0], prev_pose[1]), prev_pose[2]
            b2_entity.linearVelocity, b2_entity.angularVelocity = (0., 0.), 0.

        for pair in welded_pairs:
            self._unweld(pair)

        for pair in unwelded_pairs:
            self._weld(pair)

        if debug:
            self.display_b2world(name="After world state is reset")

    def _weld(self, pair):
        if pair not in self.b2_joints:
            body_a, body_b = self.b2_entities[pair[0]], self.b2_entities[pair[1]]
            body_b_angle_in_body_a_frame = utils.yaw_from_direction(
                body_a.GetLocalVector(utils.direction_from_yaw(body_b.angle, radians=True)), radians=True
            )
            self.b2_joints[pair] = self.b2_world.CreateWeldJoint(
                bodyA=body_a, bodyB=body_b, referenceAngle=body_b_angle_in_body_a_frame,
                localAnchorA=(0., 0.), localAnchorB=body_b.GetLocalPoint(body_a.position),
                collideConnected=False, frequencyHz=0., dampingRatio=0.
            )
            for fixture in body_a.fixtures:
                fixture.density = 1.e6
                body_a.ResetMassData()
            for fixture in body_b.fixtures:
                fixture.density = 1.e-6
                body_b.ResetMassData()

    def _unweld(self, pair):
        if pair in self.b2_joints:
            weld_joint = self.b2_joints[pair]
            self.b2_world.DestroyJoint(weld_joint)
            del self.b2_joints[pair]
            body_a, body_b = self.b2_entities[pair[0]], self.b2_entities[pair[1]]
            for fixture in body_a.fixtures:
                fixture.density = 1.
                body_a.ResetMassData()
            for fixture in body_b.fixtures:
                fixture.density = 1.
                body_b.ResetMassData()

    def deactivate_entities(self, uids):
        for uid in uids:
            if uid in self.b2_entities:
                b2_entity = self.b2_entities[uid]
                b2_entity.active = False
                self.deactivated_entities[uid] = b2_entity
                del self.b2_entities[uid]

    def activate_entities(self, uids):
        for uid in uids:
            if uid in self.deactivated_entities:
                b2_entity = self.deactivated_entities[uid]
                b2_entity.active = True
                self.b2_entities[uid] = b2_entity
                del self.deactivated_entities[uid]

    def get_entity_pose(self, uid):
        entity_body = self.b2_entities[uid]
        return entity_body.position[0], entity_body.position[1], utils.angle_to_360_interval(math.degrees(entity_body.angle))

    def query_aabb_overlapping_uids(self, polygon):
        xmin, ymin, xmax, ymax = polygon.bounds
        aabb = Box2D.b2AABB(lowerBound=(xmin, ymin), upperBound=(xmax, ymax))
        query = OverlappingEntitiesUidsQueryCallback()
        self.b2_world.QueryAABB(query, aabb)
        return query.overlapping_uids

    def b2world_to_fig(self, name=""):
        polygons_xy = [
            [
                utils.shapely_geom_to_global(
                    local_geom=Polygon(fixture.shape.vertices),
                    local_cs_pose_in_global=(body.position[0], body.position[1], math.degrees(body.angle))
                ).exterior.xy
                for fixture in body.fixtures
            ]
            for body in self.b2_world.bodies
            if body.active
        ]
        fig, ax = plt.subplots()
        for polygons in polygons_xy:
            c = numpy.random.rand(3, )
            for polygon_xy in polygons:
                ax.plot(*polygon_xy, color=c)
        ax.axis('equal')
        ax.set_title(name)
        return fig, ax

    def display_b2world(self, name="", show=False):
        fig, ax = self.b2world_to_fig(name=name)

        log_dir = "/home/xia0ben/logs/"
        existing_log_images_names = {
            name[:-4] for name in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, name))
        }
        counter = 0
        while str(counter) in existing_log_images_names:
            counter += 1

        plt.ioff()
        fig.savefig(os.path.join(log_dir, str(counter) + ".png"))

        if show:
            fig.show()
