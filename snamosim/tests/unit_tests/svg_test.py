import svgwrite  # Last update jan 15 2021
import svgutils  # Last update jan 04 2021
import svgelements  # Last update jan 18 2021


def open_svg_convert_to_shapely_and_display_in_mplt():
    with open('/home/xia0ben/INRIA/Code/s-namo-sim/data/worlds/s-namo_cases/04_after_the_feast/04_after_the_feast_complexified_4_robots.svg') as f:
        svg_data = svgelements.SVG.parse(f)
        svg_data.


if __name__ == '__main__':
    pass
