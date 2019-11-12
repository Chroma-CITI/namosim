import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely import affinity


class CircularSectorSensor:
    def __init__(self, fov_max_radius, fov_min_radius, fov_opening_angle, parent_entity_pose):
        self.fov_max_radius = fov_max_radius
        self.fov_min_radius = fov_min_radius
        self.fov_opening_angle = fov_opening_angle

        self.fov_polygon = self._create_fov(fov_max_radius, fov_min_radius, fov_opening_angle, parent_entity_pose)
        self.parent_uid = None

    def _create_fov(self, fov_max_radius, fov_min_radius, fov_opening_angle, parent_entity_pose):
        fov_outer_arc = self._create_shapely_arc(parent_entity_pose, fov_max_radius, fov_opening_angle)

        fov_inner_arc = self._create_shapely_arc(parent_entity_pose, fov_min_radius, fov_opening_angle)

        coords_outer = list(fov_outer_arc.coords)
        coords_inner = list(fov_inner_arc.coords)
        points = coords_inner + list(reversed(coords_outer))

        return Polygon(points)

    @staticmethod
    def _create_shapely_arc(entity_pose, radius, opening_angle, numsegments=15):
        start_angle, end_angle = opening_angle * -0.5, opening_angle * 0.5  # In degrees

        # The coordinates of the arc
        theta = np.radians(np.linspace(start_angle, end_angle, numsegments))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Transform in shapely
        arc_in_robot_ref = LineString(np.column_stack([x, y]))
        arc_after_trans = affinity.translate(arc_in_robot_ref, entity_pose[0], entity_pose[1])
        arc_after_trans_rot = affinity.rotate(arc_after_trans, entity_pose[2],
                                              Point(entity_pose[0], entity_pose[1]))

        return arc_after_trans_rot

    def rotate(self, angle, rot_center='centroid'):
        self.fov_polygon = affinity.rotate(self.fov_polygon, angle, rot_center)

    def translate(self, xoff, yoff):
        self.fov_polygon = affinity.translate(self.fov_polygon, xoff, yoff)
