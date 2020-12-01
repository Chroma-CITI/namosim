import unittest
from snamosim.utils import utils
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
import matplotlib.pyplot as plt
import numpy as np
import math
import Box2D


class Box2DTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_basic_with_opening(self):
        basic_polygons = {
            1: Polygon([
                (-1.3947547483983858, 1.5067685773149324), (-1.4019618999238, 1.430010414020828), (-1.4379945248779387, 1.3618531594840966), (-1.4973669460672254, 1.31267315779076), (-1.5710403014139362, 1.2899576101374424), (-1.6477984929303193, 1.2971647616628565), (-1.7159557474670506, 1.3331973583947168), (-1.7651357491603874, 1.3925698078062823), (-1.7878512968137048, 1.4662431631529929), (-1.7806441452882906, 1.5430013264470972), (-1.744611520334152, 1.6111585809838285), (-1.685239099144865, 1.6603385826771653), (-1.6115657437981543, 1.6830541303304827), (-1.5348075522817712, 1.6758470070273472), (-1.4666502977450397, 1.6398143820732085), (-1.4174702960517032, 1.580441932661643), (-1.3947547483983858, 1.5067685773149324)
            ]),  # 'robot_01'
            2: Polygon([
                (-0.022476533175288527, 2.077879660203765), (-2.142960375920752, 2.077879660203765), (-2.142960375920752, 1.9202271893432674), (-0.022476533175288527, 1.9202271893432674), (-0.022476533175288527, 2.077879660203765)
            ]),  # 'wall_top_left'
            3: Polygon([
                (2.1428570823808313, 2.0778372872745745), (1.0374827138543168, 2.0778372872745745), (1.0374827138543168, 1.9201848107696216), (2.1428570823808313, 1.9201848107696216), (2.1428570823808313, 2.0778372872745745)
            ]),  # 'wall_top_right'
            4: Polygon([
                (-0.5066130443371999, 0.8858727457454914), (-1.9780165636553495, 0.8858727457454914), (-1.9780165636553495, -1.894608698106285), (1.9701621369909406, -1.894608698106285), (1.9701621369909406, 0.8963453560240451), (0.4987587559619566, 0.8963453560240451), (0.4987587559619566, 1.0534347077583042), (2.142960375920752, 1.0534347077583042), (2.142960375920752, -2.077879660203765), (-2.1351059069229246, -2.077879660203765), (-2.1351059069229246, 1.058671012897581), (-0.5069306860835938, 1.058671012897581), (-0.5066130443371999, 0.8858727457454914)
            ]),  # 'wall_bottom'
            5: Polygon([
                (-0.5248448339118901, 1.7772249033386955), (-1.053948522563712, 1.7772249033386955), (-1.053948522563712, 1.248134225157339), (-0.5248448339118901, 1.248134225157339), (-0.5248448339118901, 1.7772249033386955)
            ])  # 'movable_box'
        }
        basic_polygons[1] = affinity.translate(basic_polygons[1], 0.3, 0.)
        # It will not be necessary to preserve these
        basic_convex_polygons = {
            uid: utils.convert_to_convex_polygons_list(polygon) for uid, polygon in basic_polygons.items()
        }

        class MyContactListener(Box2D.b2ContactListener):
            def BeginContact(self, contact):
                fix_a_uid = contact.fixtureA.body.userData['uid']
                fix_b_uid = contact.fixtureB.body.userData['uid']
                self._collision_detected = True

            def is_collision_detected(self):
                return_value = hasattr(self, '_collision_detected') and self._collision_detected
                self._collision_detected = False
                return return_value


        my_contact_listener = MyContactListener()
        box2d_world = Box2D.b2World(gravity=(0., 0.), contactListener=my_contact_listener)

        b2_bodies = {
            uid: box2d_world.CreateStaticBody(
                fixtures=[
                    Box2D.b2FixtureDef(
                        shape=Box2D.b2PolygonShape(
                            vertices=utils.coords(utils.shapely_geom_to_local(
                                polygon,
                                (basic_polygons[uid].centroid.coords[0][0], basic_polygons[uid].centroid.coords[0][1], 0.)
                            ))
                        )
                    )
                    for polygon in convex_polygons
                ],
                position=basic_polygons[uid].centroid.coords[0], angle=math.radians(0.),
                userData={'uid': uid}
            )
            for uid, convex_polygons in basic_convex_polygons.items()
            if uid != 1 and uid != 5  # Not the robot and not the obstacle
        }

        init_robot_pose = (basic_polygons[1].centroid.coords[0][0], basic_polygons[1].centroid.coords[0][1], 0.)
        robot_pose = init_robot_pose
        # robot_body = box2d_world.CreateDynamicBody(
        #     fixtures=[
        #         Box2D.b2FixtureDef(
        #             shape=Box2D.b2PolygonShape(
        #                 vertices=utils.local_shapely_polygon_coordinates(basic_polygons[1], robot_pose)
        #             )
        #         )
        #     ],
        #     position=(robot_pose[0], robot_pose[1]), angle=math.radians(robot_pose[2]), bullet=True,
        #     userData={'uid': 1}
        # )

        init_obstacle_pose = (basic_polygons[5].centroid.coords[0][0], basic_polygons[5].centroid.coords[0][1], 0.)
        obstacle_pose = init_obstacle_pose
        # obstacle_body = box2d_world.CreateDynamicBody(
        #     fixtures=[
        #         Box2D.b2FixtureDef(
        #             shape=Box2D.b2PolygonShape(
        #                 vertices=utils.local_shapely_polygon_coordinates(basic_polygons[5], obstacle_pose)
        #             )
        #         )
        #     ],
        #     position=(obstacle_pose[0], obstacle_pose[1]), angle=math.radians(obstacle_pose[2]), bullet=True,
        #     userData={'uid': 5}
        # )
        #
        # box2d_world.CreateWeldJoint(bodyA=robot_body, bodyB=obstacle_body)

        welded_polygon = MultiPolygon([basic_polygons[1], basic_polygons[5]])
        init_welded_pose = welded_polygon.centroid.coords[0][0], welded_polygon.centroid.coords[0][1], 0.
        welded_pose = init_welded_pose
        # Careful here ! if the robot or obstacle were concave, would have to use other list of polygons
        robot_local_polygon = utils.shapely_geom_to_local(basic_polygons[1], welded_pose)
        robot_local_centroid = robot_local_polygon.centroid
        robot_fixture_def = Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(vertices=utils.coords(robot_local_polygon)))
        obstacle_local_polygon = utils.shapely_geom_to_local(basic_polygons[5], welded_pose)
        obstacle_local_centroid = obstacle_local_polygon.centroid
        obstacle_fixture_def = Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(vertices=utils.coords(obstacle_local_polygon)))

        welded_body = box2d_world.CreateDynamicBody(
            fixtures=[robot_fixture_def, obstacle_fixture_def], position=(welded_pose[0], welded_pose[1]),
            angle=math.radians(welded_pose[2]), bullet=True, userData={'uid': 5}
        )

        fig, ax = plt.subplots()
        for polygon in basic_polygons.values():
            ax.plot(*polygon.exterior.xy, color='black')
        # for convex_polygons in basic_convex_polygons.values():
        #     for polygon in convex_polygons:
        #         ax.plot(*polygon.exterior.xy, color='blue')
        ax.axis('equal')
        fig.show()

        print('')

        for i in range(10):
            welded_body.linearVelocity = (0., 0.)
            welded_body.angularVelocity = math.radians(0.)

            box2d_world.Step(timeStep=1, velocityIterations=1, positionIterations=1)

            collides = my_contact_listener.is_collision_detected()

            if collides:
                welded_body.position = (init_welded_pose[0], init_welded_pose[1])
                welded_body.angle = init_welded_pose[2]
                welded_body.linearVelocity = (0., 0.)
                welded_body.angularVelocity = 0.

                box2d_world.Step(timeStep=1, velocityIterations=1, positionIterations=1)

                welded_pose = welded_body.position[0], welded_body.position[1], math.degrees(welded_body.angle)

                new_robot_centroid_coords = utils.shapely_geom_to_global(robot_local_centroid, welded_pose).coords[0]
                new_robot_pose = new_robot_centroid_coords[0], new_robot_centroid_coords[1], robot_pose[2] + welded_pose[2]
                basic_polygons[1] = utils.set_polygon_pose(basic_polygons[1], robot_pose, new_robot_pose)
                robot_pose = new_robot_pose

                new_obstacle_centroid_coords = utils.shapely_geom_to_global(obstacle_local_centroid, welded_pose).coords[0]
                new_obstacle_pose = new_obstacle_centroid_coords[0], new_obstacle_centroid_coords[1], obstacle_pose[2] + welded_pose[2]
                basic_polygons[5] = utils.set_polygon_pose(basic_polygons[5], obstacle_pose, new_obstacle_pose)
                obstacle_pose = new_obstacle_pose
            else:
                welded_pose = welded_body.position[0], welded_body.position[1], math.degrees(welded_body.angle)

                new_robot_centroid_coords = utils.shapely_geom_to_global(robot_local_centroid, welded_pose).coords[0]
                new_robot_pose = new_robot_centroid_coords[0], new_robot_centroid_coords[1], robot_pose[2] + welded_pose[2]
                basic_polygons[1] = utils.set_polygon_pose(basic_polygons[1], robot_pose, new_robot_pose)
                robot_pose = new_robot_pose

                new_obstacle_centroid_coords = utils.shapely_geom_to_global(obstacle_local_centroid, welded_pose).coords[0]
                new_obstacle_pose = new_obstacle_centroid_coords[0], new_obstacle_centroid_coords[1], obstacle_pose[2] + welded_pose[2]
                basic_polygons[5] = utils.set_polygon_pose(basic_polygons[5], obstacle_pose, new_obstacle_pose)
                obstacle_pose = new_obstacle_pose

            fig, ax = plt.subplots()
            for polygon in basic_polygons.values():
                ax.plot(*polygon.exterior.xy, color='black')
            ax.axis('equal')
            fig.show()

        print('')


if __name__ == '__main__':
    unittest.main()
