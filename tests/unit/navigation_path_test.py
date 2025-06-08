import unittest

import numpy as np
from shapely.geometry import Polygon

import namosim.navigation.basic_actions as ba
from namosim.navigation.navigation_path import TransitPath
from namosim.world.entity import Movability, Style
from namosim.world.obstacle import Obstacle


class TestTransitPathFromPoses:
    def setup_method(self):
        self.simple_square = Obstacle(
            uid="simple_square",
            polygon=Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            pose=(0.0, 0.0, 0.0),
            type_="box",
            style=Style(),
            movability=Movability.MOVABLE,
        )

    def test_from_poses(self):
        robot_pose = (401.81603000000007, -184.19798500000007, 90.0)
        poses = [
            (401.81603000000007, -184.19798500000007, 90.0),
            (413.3967763298543, -184.4751680398556, -129.75446123154683),
        ]
        path = TransitPath.from_poses(
            robot_pose=robot_pose,
            robot_polygon=Polygon([(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0)]),
            poses=poses,
        )
        assert len(path.actions) == 3
        assert isinstance(path.actions[0], ba.Rotation)
        assert isinstance(path.actions[1], ba.Advance)
        assert isinstance(path.actions[2], ba.Rotation)
        pose = robot_pose
        for action in path.actions:
            if isinstance(action, ba.Advance):
                pose = action.predict_pose(pose, pose)
            elif isinstance(action, ba.Rotation):
                pose = action.predict_pose(pose, pose)

        assert np.allclose(poses[-1][:2], pose[:2], rtol=1e-6)

    def test_no_actions(self):
        path = TransitPath.from_poses(
            robot_pose=(0, 0, 0),
            robot_polygon=Polygon([(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0)]),
            poses=[
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
            ],
        )
        assert len(path.actions) == 0
        assert len(path.robot_path.poses) == 1

    def test_bug_case(self):
        robot_polygon = Polygon(
            [
                (-167.8543151399452, 28.17133002255224),
                (-157.26338514008393, 27.176910022797415),
                (-147.8591951402002, 22.205190022185832),
                (-141.0734351397441, 14.013120022451574),
                (-137.93918514071277, 3.84783002232831),
                (-138.93360514034828, -6.743089978067076),
                (-143.90532514096003, -16.147279977950802),
                (-152.09739514056758, -22.933039977415035),
                (-162.2626851408175, -26.06728997743835),
                (-172.85360514108618, -25.072869976810694),
                (-182.25779514109655, -20.101149977191085),
                (-189.0435551404342, -11.909079976591556),
                (-192.1778051405841, -1.743789977333563),
                (-191.1833851399565, 8.847130022935119),
                (-186.21166514033678, 18.251320022945492),
                (-178.0195951396106, 25.03708002327511),
                (-167.8543151399452, 28.17133002255224),
            ]
        )
        robot_pose = (-165.05849501368272, 1.052019999999942, 0.0)
        poses = [
            (-165.05849501368272, 1.052019999999942, 0.0),
            (-165.90581999999998, 4.886479993158559, 90.0),
        ]

        path = TransitPath.from_poses(
            robot_pose=robot_pose, robot_polygon=robot_polygon, poses=poses
        )

        # Apply the actions and make sure they take us to the goal pose
        pose = robot_pose
        for action in path.actions:
            if isinstance(action, ba.Advance):
                pose = action.predict_pose(pose, pose)
            elif isinstance(action, ba.Rotation):
                pose = action.predict_pose(pose, pose)

        assert np.allclose(poses[-1], pose, rtol=1e-6)

    def test_bug_case_2(self):
        robot_pose = (-202.5176508239593, 342.66973604540215, 60.0)
        poses = [
            (-202.5176508239593, 342.66973604540215, 60.0),
            (-202.5176508239593, 342.66973604540215, 8.455055627136348),
            (-182.49998499999998, 345.64534499999996, 8.455055627136348),
            (-182.49998499999998, 345.64534499999996, 180.0),
            (-182.499985, 345.64534499999996, 180.0),
            (-182.499985, 345.64534499999996, 127.37593160586596),
        ]

        path = TransitPath.from_poses(
            robot_pose=robot_pose,
            robot_polygon=Polygon([(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0)]),
            poses=poses,
        )

        # Apply the actions and make sure they take us to the goal pose
        pose = robot_pose
        for action in path.actions:
            if isinstance(action, ba.Advance):
                pose = action.predict_pose(pose, pose)
            elif isinstance(action, ba.Rotation):
                pose = action.predict_pose(pose, pose)

        assert np.allclose(poses[-1], pose, rtol=1e-6)
