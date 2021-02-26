from snamosim.utils import utils
import math
import Box2D
import snamosim.behaviors.plan.basic_actions as ba

import matplotlib.pyplot as plt
from shapely.geometry import Polygon


class MyContactListener(Box2D.b2ContactListener):
    def BeginContact(self, contact):
        # TODO save collision pairs because there can be several of these
        self._collision_detected = True
        # self.fix_a_uid = contact.fixtureA.userData['uid']
        # self.fix_b_uid = contact.fixtureB.userData['uid']

    def is_collision_detected(self):
        return_value = hasattr(self, '_collision_detected') and self._collision_detected
        self._collision_detected = False
        return return_value


class B2Sim:
    def __init__(self, other_entities_polygons, other_entities_poses, obstacle_polygon, obstacle_pose,
                 robot_polygons, robot_poses, robot_uid, obstacle_uid):

        # Save initial parameters
        self._other_entities_polygons, self._other_entities_poses = other_entities_polygons, other_entities_poses
        self._obstacle_polygon, self._obstacle_pose = obstacle_polygon, obstacle_pose
        self._robot_polygons, self._robot_poses = robot_polygons, robot_poses
        self._robot_uid, self._obstacle_uid = robot_uid, obstacle_uid

        # Initialize box2d world
        self.contact_listener = MyContactListener()
        self.box2d_world = Box2D.b2World(gravity=(0., 0.), contactListener=self.contact_listener)

        # Init static obstacles as static bodies
        self.b2_bodies = {}
        for uid, polygon in other_entities_polygons.items():
            local_polygon = utils.shapely_geom_to_local(polygon, other_entities_poses[uid])
            convex_polygons_coords = utils.convert_to_convex_polygons_coordinates_list(local_polygon)
            fixtures = []
            for counter, coords in enumerate(convex_polygons_coords):
                try:
                    fixture = Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(vertices=coords), userData={'uid': uid})
                    fixtures.append(fixture)
                except Exception as e:
                    # TODO: Either improve convexity verification to recognize after the feast tables as convex
                    #  or fix after the feast case to stop using rounded corners that are probably causing the problem
                    #  when svg is read and simplified to polygon.
                    pass

            self.b2_bodies[uid] = self.box2d_world.CreateStaticBody(
                fixtures=fixtures,
                position=(other_entities_poses[uid][0], other_entities_poses[uid][1]),
                angle=other_entities_poses[uid][2]
            )

        # Init robot and moved obstacle as a single dynamic body, for each manipulation pose
        self.manip_pose_id_to_welded_body = {}
        for manip_pose_id, robot_polygon in robot_polygons.items():
            robot_pose = robot_poses[manip_pose_id]

            local_robot_polygon = utils.shapely_geom_to_local(robot_polygon, robot_pose)
            robot_convex_polygons_coords = utils.convert_to_convex_polygons_coordinates_list(local_robot_polygon)
            robot_fixtures_defs = [
                Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(vertices=coords), userData={'uid': robot_uid})
                for coords in robot_convex_polygons_coords
            ]

            local_obstacle_polygon = utils.shapely_geom_to_local(obstacle_polygon, robot_pose)
            obstacle_convex_polygons_coords = utils.convert_to_convex_polygons_coordinates_list(local_obstacle_polygon)
            obstacle_fixtures_defs = [
                Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(vertices=coords), userData={'uid': obstacle_uid})
                for coords in obstacle_convex_polygons_coords if len(coords) > 3 # TODO This extra verification should not be necessary but whatever
            ]

            self.manip_pose_id_to_welded_body[manip_pose_id] = self.box2d_world.CreateDynamicBody(
                fixtures=robot_fixtures_defs + obstacle_fixtures_defs, position=(robot_pose[0], robot_pose[1]),
                angle=math.radians(robot_pose[2]), bullet=True,
                userData={'obstacle_local_centroid': local_obstacle_polygon.centroid},
                active=False  # Start out as non-enabled (only enabled when used). Note: 'enabled' in v2.4.x
            )

    def display_b2world(self):
        polygons_xy = [
            utils.shapely_geom_to_global(
                local_geom=Polygon(fixture.shape.vertices),
                local_cs_pose_in_global=(body.position[0], body.position[1], math.degrees(body.angle))
            ).exterior.xy
            for body in self.box2d_world.bodies for fixture in body.fixtures
            if body.active
        ]
        fig, ax = plt.subplots()
        for polygon_xy in polygons_xy:
            ax.plot(*polygon_xy, color='black')
        ax.axis('equal')
        fig.show()

    def check_action_collides(self, robot_pose, manip_pose_id, action, debug_init=False, debug_after=False):
        # Setup velocities according to action
        welded_body = self.manip_pose_id_to_welded_body[manip_pose_id]
        welded_body.active = True
        welded_body.position, welded_body.angle = (robot_pose[0], robot_pose[1]), math.radians(robot_pose[2])
        if isinstance(action, ba.Translation):
            welded_body.linearVelocity, welded_body.angularVelocity = action.compute_translation_vector(robot_pose), 0.
        if isinstance(action, ba.Rotation):
            welded_body.linearVelocity, welded_body.angularVelocity = (0., 0.), math.radians(action.angle)

        if debug_init:
            self.display_b2world()

        # Have Box2D simulate the action
        self.box2d_world.Step(timeStep=1, velocityIterations=1, positionIterations=1)

        if debug_after:
            self.display_b2world()

        welded_body.active = False

        collides = self.contact_listener.is_collision_detected()

        return collides

    def get_polygons_after_action(self, robot_pose, manip_pose_id, action):
        # Setup velocities according to action
        welded_body = self.manip_pose_id_to_welded_body[manip_pose_id]
        welded_body.active = True
        welded_body.position, welded_body.angle = (robot_pose[0], robot_pose[1]), math.radians(robot_pose[2])
        if isinstance(action, ba.Translation):
            welded_body.linearVelocity, welded_body.angularVelocity = action.compute_translation_vector(robot_pose), 0.
        if isinstance(action, ba.Rotation):
            welded_body.linearVelocity, welded_body.angularVelocity = (0., 0.), math.radians(action.angle)

        self.box2d_world.Step(timeStep=1, velocityIterations=1, positionIterations=1)

        welded_body.active = False

        new_robot_pose = (
            welded_body.position[0], welded_body.position[1],
            utils.angle_to_360_interval(math.degrees(welded_body.angle))
        )
        new_robot_polygon = utils.set_polygon_pose(
            self._robot_polygons[manip_pose_id], self._robot_poses[manip_pose_id], new_robot_pose)

        new_obstacle_centroid_coords = utils.shapely_geom_to_global(
            welded_body.userData['obstacle_local_centroid'], new_robot_pose
        ).coords[0]
        new_obstacle_pose = (
            new_obstacle_centroid_coords[0], new_obstacle_centroid_coords[1],
            utils.angle_to_360_interval(
                self._obstacle_pose[2] + new_robot_pose[2] - self._robot_poses[manip_pose_id][2]
            )
        )
        new_obstacle_polygon = utils.set_polygon_pose(self._obstacle_polygon, self._obstacle_pose, new_obstacle_pose)

        return new_robot_polygon, new_obstacle_polygon
