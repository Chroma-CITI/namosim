import unittest
from snamosim.utils import utils
from shapely.geometry import Polygon
from shapely import affinity
import matplotlib.pyplot as plt
import math
import Box2D


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


class Box2DTest(unittest.TestCase):
    def setUp(self):
        pass

    @staticmethod
    def initialize_box_2d_world(polygons, poses, robot_uid, obstacle_uid):
        contact_listener = MyContactListener()
        box2d_world = Box2D.b2World(gravity=(0., 0.), contactListener=contact_listener)

        b2_bodies = {}

        for uid, polygon in polygons.items():
            if uid != robot_uid and uid != obstacle_uid:
                convex_polygons_coords = utils.convert_to_convex_polygons_coordinates_list(
                    utils.shapely_geom_to_local(polygon, poses[uid])
                )

                b2_bodies[uid] = box2d_world.CreateStaticBody(
                    fixtures=[
                        Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(vertices=coords), userData={'uid': uid})
                        for coords in convex_polygons_coords
                    ],
                    position=(poses[uid][0], poses[uid][1]), angle=poses[uid][2]
                )

        local_robot_polygon = utils.shapely_geom_to_local(polygons[robot_uid], poses[robot_uid])
        robot_convex_polygons_coords = utils.convert_to_convex_polygons_coordinates_list(local_robot_polygon)
        robot_fixtures_defs = [
            Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(vertices=coords), userData={'uid': robot_uid})
            for coords in robot_convex_polygons_coords
        ]

        local_obstacle_polygon = utils.shapely_geom_to_local(polygons[obstacle_uid], poses[robot_uid])
        obstacle_convex_polygons_coords = utils.convert_to_convex_polygons_coordinates_list(local_obstacle_polygon)
        obstacle_fixtures_defs = [
            Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(vertices=coords), userData={'uid': obstacle_uid})
            for coords in obstacle_convex_polygons_coords
        ]

        welded_body = box2d_world.CreateDynamicBody(
            fixtures=robot_fixtures_defs + obstacle_fixtures_defs, position=(poses[robot_uid][0], poses[robot_uid][1]),
            angle=math.radians(poses[robot_uid][2]), bullet=True,
            userData={'obstacle_local_centroid': local_obstacle_polygon.centroid}
        )

        return box2d_world, b2_bodies, welded_body, contact_listener

    def test_basic_with_opening(self):
        robot_uid, obstacle_uid = 1, 5
        polygons = {
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
        polygons[robot_uid] = affinity.translate(polygons[robot_uid], 0.3, 0.)
        poses = {
            uid: (polygon.centroid.coords[0][0], polygon.centroid.coords[0][1], 0.) for uid, polygon in polygons.items()
        }

        box2d_world, b2_bodies, welded_body, contact_listener = self.initialize_box_2d_world(
            polygons, poses, robot_uid, obstacle_uid
        )

        init_welded_body_position, init_welded_body_angle = tuple(welded_body.position), welded_body.angle
        init_robot_pose, init_obstacle_pose = poses[robot_uid], poses[obstacle_uid]

        fig, ax = plt.subplots()
        for polygon in polygons.values():
            ax.plot(*polygon.exterior.xy, color='black')
        ax.axis('equal')
        fig.show()

        print('')

        for i in range(2):
            welded_body.linearVelocity, welded_body.angularVelocity = (0., 0.), math.radians(180.)

            box2d_world.Step(timeStep=1, velocityIterations=1, positionIterations=1)

            collides = contact_listener.is_collision_detected()

            if collides:
                welded_body.position, welded_body.angle = init_welded_body_position, init_welded_body_angle
                welded_body.linearVelocity, welded_body.angularVelocity = (0., 0.), 0.

                box2d_world.Step(timeStep=1, velocityIterations=1, positionIterations=1)

            new_robot_pose = (
                welded_body.position[0], welded_body.position[1],
                utils.angle_to_360_interval(math.degrees(welded_body.angle))
            )
            polygons[robot_uid] = utils.set_polygon_pose(polygons[robot_uid], poses[robot_uid], new_robot_pose)
            poses[robot_uid] = new_robot_pose

            new_obstacle_centroid_coords = utils.shapely_geom_to_global(
                welded_body.userData['obstacle_local_centroid'], new_robot_pose
            ).coords[0]
            new_obstacle_pose = (
                new_obstacle_centroid_coords[0], new_obstacle_centroid_coords[1],
                utils.angle_to_360_interval(init_obstacle_pose[2] + new_robot_pose[2] - init_robot_pose[2])
            )
            polygons[obstacle_uid] = utils.set_polygon_pose(
                polygons[obstacle_uid], poses[obstacle_uid], new_obstacle_pose
            )
            poses[obstacle_uid] = new_obstacle_pose

            fig, ax = plt.subplots()
            for polygon in polygons.values():
                ax.plot(*polygon.exterior.xy, color='black')
            ax.axis('equal')
            fig.show()

            print('')


if __name__ == '__main__':
    unittest.main()
s