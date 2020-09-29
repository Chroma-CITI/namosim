import unittest
from src.utils.collision import *


class CollisionTest(unittest.TestCase):

    def setUp(self):
        action_list = ActionList([
            LinearMovement(translation_vector=(0.4, 1.1), angle=45., center='center'),
            LinearMovement(translation_vector=(0.6, 1.0), angle=20., center='center')
        ])
        self.moved_polygon = Polygon([(0., 0.), (0., 1.), (1., 1.), (1., 0.75), (0.25, 0.75), (0.25, 0)])

        self.moved_polygon_states = action_list.apply(self.moved_polygon)

        self.polygon_that_should_collide = Polygon([(1.45, 1.75), (1.45, 1.25), (1.95, 1.25), (1.95, 1.75)])
        self.polygon_that_should_not_collide = affinity.translate(self.polygon_that_should_collide, 1., 1.)

        _fig, _ax = self.make_base_polygon_figures()

        # _bb_vertices = bounding_boxes_vertices(self.moved_polygon_states, action_list.actions, bb_type='aabbox')
        # _x, _y = zip(*[[_vertex.x, _vertex.y] for _vertex in _bb_vertices])
        # _ax.scatter(_x, _y)
        #
        # csv = csv_from_bb_vertices(_bb_vertices)
        # _ax.plot(*csv.exterior.xy)
        #
        # # Test circle finding method
        # poly_mi_2 = affinity.translate(affinity.rotate(self.moved_polygon, 22.5), 0.2, 0.55)
        # _ax.plot(*poly_mi_2.exterior.xy)
        #
        # synced_coords = zip(
        #     list(self.moved_polygon.exterior.coords),
        #     list(poly_mi_2.exterior.coords),
        #     list(self.moved_polygon_states[1].exterior.coords)
        # )
        #
        # circles_terms = [
        #     utils.find_circle_terms(
        #         coords[0][0], coords[0][1],
        #         coords[1][0], coords[1][1],
        #         coords[2][0], coords[2][1]
        #     )
        #     for coords in synced_coords
        # ]
        #
        # plt_circles = [plt.Circle((circle_terms[0], circle_terms[1]), circle_terms[2], fill=False) for circle_terms in
        #                circles_terms]
        #
        # for circle in plt_circles:
        #     _ax.add_artist(circle)
        # # _ax.add_artist(plt_circles[1])
        #
        # angles = [
        #     (
        #         utils.points_to_angle(
        #             coords[0][0], coords[0][1], circle_terms[0], circle_terms[1], coords[1][0], coords[1][1]
        #         ),
        #         utils.points_to_angle(
        #             coords[1][0], coords[1][1], circle_terms[0], circle_terms[1], coords[2][0], coords[2][1]
        #         )
        #     )
        #     for circle_terms, coords in zip(circles_terms, synced_coords)
        # ]
        #
        # deg_angles = [math.degrees(angle_1 + angle_2) for angle_1, angle_2 in angles]
        #
        # # Display everything
        # _ax.axis('equal')
        # # plt.axhline(y=0)
        # # plt.axvline(x=0)
        # _fig.show()
        #
        # obs_collides, obs_collision_data, _aabb_tree = csv_check_collisions(
        #     [self.polygon_that_should_collide], self.moved_polygon_states, action_list.actions, display_debug=True)
        # print(str(obs_collides))
        # not_obs_collides, not_obs_collision_data, not_obs_aabb_tree = csv_check_collisions(
        #     [self.polygon_that_should_not_collide], self.moved_polygon_states, action_list.actions, display_debug=True)
        # print(str(not_obs_collides))

    def make_base_polygon_figures(self):
        _fig, _ax = plt.subplots()
        for p in self.moved_polygon_states:
            _ax.plot(*p.exterior.xy)
        _ax.plot(*self.polygon_that_should_collide.exterior.xy)
        _ax.plot(*self.polygon_that_should_not_collide.exterior.xy)
        return _fig, _ax

    def test_visualize(self):
        pass

    def test_arc_bounding_box(self):
        test_poly_state_1_coords = [(0., 1.), (0., 2.), (1., 2.), (1., 1.)]
        test_poly_state_1 = Polygon(test_poly_state_1_coords)
        rotation = Rotation(angle=190., center=(0.5, 0.5))
        test_poly_state_2 = rotation.apply(test_poly_state_1)
        test_poly_state_2_coords = list(test_poly_state_2.exterior.coords)

        bbs_and_construction_points = [
            arc_bounding_box(
                point_a=test_poly_state_1_coords[index], point_b=test_poly_state_2_coords[index],
                rot_angle=rotation.angle, center=rotation.center, trans_vector=None
            )
            for index in range(4)
        ]

        _fig, _ax = plt.subplots()
        _ax.plot(*test_poly_state_1.exterior.xy, color='black')
        _ax.plot(*test_poly_state_2.exterior.xy, color='blue')

        # _ax.plot(*bbs[2].exterior.xy, color='red')

        for bb, construction_points in bbs_and_construction_points:
            _ax.plot(*Polygon(bb).exterior.xy, color='red')
            _ax.scatter(*zip(*construction_points), color='black')

        # for bb_points, construction_points in bbs_and_construction_points:
        #     if hasattr(Polygon().exterior, 'xy'):
        #         _ax.plot(*bb.exterior.xy, color='red')

        # _ax.plot(*affinity.rotate(test_poly_state_1, angle=45., origin=(0.5, 0.5)).exterior.xy, color='green')
        # _ax.plot(*affinity.rotate(test_poly_state_1, angle=135., origin=(0.5, 0.5)).exterior.xy, color='green')

        _ax.axis('equal')
        _fig.show()
        print('')

    def test_


if __name__ == '__main__':
    unittest.main()