import copy
import math
import Box2D
import os
import json
from Box2D.examples.framework import (Framework, Keys, main)
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2_dynamicBody,
                   b2_kinematicBody, b2_staticBody)
from snamosim.utils import utils
import snamosim.behaviors.plan.basic_actions as ba
from snamosim.worldreps.entity_based.world import World


class CollisionPairsContactListener(Box2D.b2ContactListener):
    def __init__(self, **kwargs):
        Box2D.b2ContactListener.__init__(self, **kwargs)
        self._contacts = {}

    def BeginContact(self, contact):
        self._contacts[(contact.fixtureA.userData['uid'], contact.fixtureB.userData['uid'])] = contact
        # print(self.get_collision_pairs())

    def EndContact(self, contact):
        try:
            del self._contacts[(contact.fixtureA.userData['uid'], contact.fixtureB.userData['uid'])]
        except KeyError:
            pass
        # print(self.get_collision_pairs())

    def get_collides_with(self, ignored_collision_pairs=None):
        if ignored_collision_pairs:
            collision_pairs = set(self._contacts.keys()).difference(ignored_collision_pairs)
        else:
            collision_pairs = self._contacts.keys()
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


class OverlappingEntitiesUidsQueryCallback(Box2D.b2QueryCallback):
    def __init__(self):
        Box2D.b2QueryCallback.__init__(self)
        self.overlapping_uids = set()

    def ReportFixture(self, fixture):
        self.overlapping_uids.add(fixture.userData['uid'])
        return True # Continue the query by returning True


class BodyTypes(Framework):
    name = "Body Types"
    description = "Change body type keys: (d) dynamic, (s) static, (k) kinematic"

    def __init__(self):
        self.contact_listener = CollisionPairsContactListener()

        super(BodyTypes, self).__init__()

        # self.world.contactListener = self.contact_listener
        self.world.gravity = (0., 0.)
        self.entities = {}
        self.joints = {}

        simulation_file_abs_path = '/home/xia0ben/INRIA/Code/s-namo-sim/data/simulations/s-namo_cases/02_basic_with_opening/stilman_2005_behavior.json'
        with open(simulation_file_abs_path) as f:
            self.config = json.load(f)
        world_file_path = self.config["files"]["world_file"]
        world_abs_path = os.path.join(os.path.dirname(simulation_file_abs_path), world_file_path)
        self.init_ref_world = World.load_from_json(world_abs_path)

        self.add_entities(self.init_ref_world.entities)

        body_b_angle_in_body_a_frame = utils.yaw_from_direction(
            self.entities[1].GetLocalVector(
                utils.direction_from_yaw(self.entities[5].angle, radians=True)), radians=True
        )
        self.joints[(1, 5)] = self.world.CreateWeldJoint(
            bodyA=self.entities[1],
            bodyB=self.entities[5],
            collideConnected=False,
            localAnchorA=(0., 0.),
            localAnchorB=self.entities[5].GetLocalPoint(self.entities[1].position),
            referenceAngle=body_b_angle_in_body_a_frame,
            frequencyHz=0.,
            dampingRatio=0.
        )
        for fixture in self.entities[1].fixtures:
            fixture.density=1e7
            self.entities[1].ResetMassData()
        for fixture in self.entities[5].fixtures:
            fixture.density=1e-7
            self.entities[1].ResetMassData()

        pass
        # self.settings.hz = 1./5.
        # self.settings.positionIterations = 1
        # self.settings.velocityIterations = 1

    def Keyboard(self, key):
        pass
        if key == Keys.K_w:
            self.entities[1].linearVelocity = 0., 0.
        elif key == Keys.K_s:
            self.entities[1].linearVelocity = self.entities[1].linearVelocity[0], -5.
        elif key == Keys.K_d:
            self.entities[1].linearVelocity = 5., self.entities[1].linearVelocity[1]

    def Step(self, settings):
        super(BodyTypes, self).Step(self.settings)

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
            self.entities[uid] = self.world.CreateBody(
                defn=Box2D.b2BodyDef(
                    type=2,
                    position=(entity.pose[0], entity.pose[1]),
                    angle=math.radians(entity.pose[2]),
                    linearVelocity=(0., 0.),
                    angularVelocity=0.,
                    linearDamping=0.,
                    angularDamping=0.,
                    allowSleep=True,
                    awake=True,
                    fixedRotation=False,
                    bullet=True,
                    active=True,
                    gravityScale=1.,
                    userData={'uid': uid}
                )
            )
            for fixture_def in self.polygon_to_fixtures_defs(entity.polygon, entity.pose, uid):
                self.entities[uid].CreateFixture(defn=fixture_def)

    def remove_entities(self, entities_uids):
        for uid in entities_uids:
            self.world.DestroyBody(self.entities[uid])
            del self.entities[uid]

    def update_entities(self, entities, debug_init=False, debug_after=False):
        if debug_init:
            self.display_b2world()

        for uid, entity in entities.items():
            self.entities[uid].position = (entity.pose[0], entity.pose[1])
            self.entities[uid].angle = math.radians(entity.pose[2])

        # Have Box2D do one step to ensure aabbtree is updated
        self.world.Step(
            timeStep=self.time_step,
            velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
        )

        if debug_after:
            self.display_b2world()

    def simulate_simple_kinematics(self, list_of_uid_to_actions, init_poses=None, init_welded_pairs=None,
                                   apply=False, debug_all=False, debug_before_init=False, debug_after_init=False,
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
        init_poses = init_poses if init_poses is not None else dict()
        init_welded_pairs = init_welded_pairs if init_welded_pairs is not None else set()

        previous_poses = self._initialize_world_state(
            init_poses, init_welded_pairs, debug=debug_all or debug_before_init
        )

        # Have Box2D apply the initial state and add collisions if there are some
        self.world.Step(
            timeStep=self.time_step,
            velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
        )

        collides_with = self.contact_listener.get_collides_with(init_welded_pairs)

        if debug_all or debug_after_init:
            self.display_b2world(name="After initial world state is set")

        if collides_with:
            if not apply:
                self._reset_world_state(previous_poses, init_welded_pairs, debug_all or debug_after_reset)
            else:
                # If apply, at least reset the colliding entities
                colliding_uids = set(collides_with.keys())
                self._reset_world_state(
                    {uid: pose for uid, pose in previous_poses if uid in colliding_uids},
                    {pair for pair in init_welded_pairs if pair[0] in colliding_uids or pair[1] in colliding_uids},
                    debug_all or debug_after_reset
                )
            return collides_with

        # Try to apply actions
        nb_actions = len(list_of_uid_to_actions)
        for action_index, uid_to_action in enumerate(list_of_uid_to_actions):
            ignored_collision_pairs = set()
            for uid, action in uid_to_action.items():
                b2_entity = self.entities[uid]
                if isinstance(action, ba.Translation):
                    if isinstance(action, (ba.Grab, ba.Release)):
                        pair, reverse_pair = (uid, action.entity_uid), (action.entity_uid, uid)
                        ignored_collision_pairs.add(pair)
                        ignored_collision_pairs.add(reverse_pair)
                        if isinstance(action, ba.Release):
                            if pair in self.joints:
                                self._unweld(pair)
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
                self.world.Step(
                    timeStep=self.time_step,
                    velocityIterations=self.velocity_iterations, positionIterations=self.position_iterations
                )

            # Reset velocities and apply weld joint for grab actions
            for uid, action in uid_to_action.items():
                b2_entity = self.entities[uid]
                b2_entity.linearVelocity, b2_entity.angularVelocity = (0., 0.), 0.
                if isinstance(action, ba.Grab):
                    pair = (uid, action.entity_uid)
                    if pair not in self.joints:
                        self._weld(pair)

            collides_with = self.contact_listener.get_collides_with(init_welded_pairs.union(ignored_collision_pairs))

            if debug_all or debug_after_each_step:
                self.display_b2world(name="After applying step {}/{}".format(action_index + 1, nb_actions))
                pass

            if collides_with:
                if not apply:
                    self._reset_world_state(previous_poses, init_welded_pairs, debug_all or debug_after_reset)
                else:
                    # If apply, at least reset the colliding entities
                    colliding_uids = set(collides_with.keys())
                    self._reset_world_state(
                        {uid: pose for uid, pose in previous_poses if uid in colliding_uids},
                        {pair for pair in init_welded_pairs if pair[0] in colliding_uids or pair[1] in colliding_uids},
                        debug_all or debug_after_reset
                    )
                return collides_with

        # Reset world state
        if not apply:
            self._reset_world_state(previous_poses, init_welded_pairs, debug_all or debug_after_reset)

        return collides_with  # Should be an empty dict

    def _initialize_world_state(self, init_poses, init_welded_pairs, debug=False):
        if debug:
            self.display_b2world(name="Before initial world state is set")

        previous_poses = {}  # CAREFUL, ANGLES ARE SAVED IN RADIANS !
        for uid, pose in init_poses.items():
            b2_entity = self.entities[uid]
            previous_poses[uid] = (b2_entity.position[0], b2_entity.position[1], b2_entity.angle)
            b2_entity.position, b2_entity.angle = (pose[0], pose[1]), math.radians(pose[2])

        for pair in init_welded_pairs:
            self._weld(pair)

        return previous_poses

    def _reset_world_state(self, previous_poses, welded_pairs, debug=False):
        for uid, prev_pose in previous_poses.items():
            b2_entity = self.entities[uid]
            b2_entity.position, b2_entity.angle = (prev_pose[0], prev_pose[1]), prev_pose[2]
            b2_entity.linearVelocity, b2_entity.angularVelocity = (0., 0.), 0.

        for pair in welded_pairs:
            self._unweld(pair)

        if debug:
            self.display_b2world(name="After world state is reset")

    def _weld(self, pair):
        body_a, body_b = self.entities[pair[0]], self.entities[pair[1]]
        body_b_angle_in_body_a_frame = utils.yaw_from_direction(
            body_a.GetLocalVector(utils.direction_from_yaw(body_b.angle, radians=True)), radians=True
        )
        self.joints[pair] = self.world.CreateWeldJoint(
            bodyA=body_a, bodyB=body_b, referenceAngle=body_b_angle_in_body_a_frame,
            localAnchorA=tuple(body_a.position), localAnchorB=body_b.GetLocalPoint(body_a.position),
            collideConnected=False, frequencyHz=0., dampingRatio=0.
        )
        for fixture in body_a.fixtures:
            fixture.density = 1.e6
        for fixture in body_b.fixtures:
            fixture.density = 1.e-6

    def _unweld(self, pair):
        weld_joint = self.joints[pair]
        self.world.DestroyJoint(weld_joint)
        del self.joints[pair]
        body_a, body_b = self.entities[pair[0]], self.entities[pair[1]]
        for fixture in body_a.fixtures:
            fixture.density = 1.
        for fixture in body_b.fixtures:
            fixture.density = 1.


if __name__ == "__main__":
    main(BodyTypes)
