from entity import Entity
import shapely.affinity as affinity


class Robot(Entity):

    def __init__(self, name, polygon, dd, radius, g_fov_polygon, s_fov_polygon, initial_pose):
        Entity.__init__(self, name, polygon, dd)
        self.radius = radius
        self.g_fov_polygon = g_fov_polygon
        self.s_fov_polygon = s_fov_polygon
        self.pose = initial_pose

    def rotate(self, angle, dd):
        Entity.rotate(self, angle, dd)
        self.g_fov_polygon = affinity.rotate(self.g_fov_polygon, angle, (self.pose[0], self.pose[1]))
        self.s_fov_polygon = affinity.rotate(self.s_fov_polygon, angle, (self.pose[0], self.pose[1]))

    def translate(self, xoff, yoff, dd):
        Entity.translate(self, xoff, yoff, dd)
        self.g_fov_polygon = affinity.translate(self.g_fov_polygon, xoff, yoff)
        self.s_fov_polygon = affinity.translate(self.s_fov_polygon, xoff, yoff)
