import rospy
from ros_publisher import RosPublisher
from shapely.geometry import Polygon
from shapely.ops import triangulate
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Quaternion, Point, Vector3
from std_msgs.msg import Header, ColorRGBA

rospy.init_node('world_gui_test_node', log_level=rospy.INFO)

# Create ros_publisher

rp = RosPublisher("/home/xia0ben/catkin_ws/src/namo_navigation/config/config.yaml")

# polygon=Polygon([[-0.95, 2.0], [-0.95, 0.0], [0.95, 0.0], [2.0, 2.0], [0.95, 2.0]])
polygon = Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

triangles = triangulate(polygon)


marker = Marker(type=Marker.TRIANGLE_LIST,
                ns='/polygons',
                id=0,
                header=Header(frame_id='map'),
                color=ColorRGBA(0.0, 1.0, 0.0, 1.0),
                scale=Vector3(1.0, 1.0, 1.0),
                points=[])
for triangle in triangles:
    for point in triangle.exterior.coords[:len(triangle.exterior.coords) - 1]:
        marker.points.append(Point(point[0], point[1], 0.0))

marker_line_strip = Marker(type=Marker.LINE_STRIP,
                           ns='/lines',
                           id=0,
                           header=Header(frame_id="map"),
                           color=ColorRGBA(0.0, 0.0, 1.0, 1.0),
                           scale=Vector3(0.05, 0.0, 0.0),
                           points=[])
for i in range(len(polygon.exterior.coords) - 1):
    point = polygon.exterior.coords[i]
    next_point = polygon.exterior.coords[i+1]
    marker_line_strip.points.append(Point(point[0], point[1], 0.0))
    marker_line_strip.points.append(Point(next_point[0], next_point[1], 0.0))
marker_line_strip.points.append(Point(polygon.exterior.coords[0][0], polygon.exterior.coords[0][1], 0.0))
marker_line_strip.points.append(Point(polygon.exterior.coords[1][0], polygon.exterior.coords[1][1], 0.0))


text = "Name: " + "O1" + "\n" + "Type: " + "Box" + "\n" + "Movability: " + "Movable" + "\n" + "UID: " + "1"

marker_text = Marker(type=Marker.TEXT_VIEW_FACING,
                     ns='/polygons/text',
                     id=0,
                     pose=Pose(Point(polygon.centroid.coords[0][0], polygon.centroid.coords[0][1], 0.0), Quaternion()),
                     scale=Vector3(0.0, 0.0, 0.1),
                     header=Header(frame_id='map'),
                     color=ColorRGBA(0.0, 0.0, 0.0, 1.0),
                     text=text)

marker_array = MarkerArray()
marker_array.markers = [marker, marker_line_strip, marker_text]

rp.publish('visualization_marker_array', marker_array)

rate = rospy.Rate(0.25)
rate.sleep()

marker = Marker(ns='/polygons',
                id=0,
                action=Marker.DELETE)

marker_line_strip = Marker(ns='/lines',
                           id=0,
                           action=Marker.DELETE)

marker_array = MarkerArray()
marker_array.markers = [marker, marker_line_strip]
rp.publish('visualization_marker_array', marker_array)
