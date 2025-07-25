import colorsys
import typing as t

from std_msgs.msg import ColorRGBA


def hex_to_rgba(hex_string_in: str):
    hex_string = hex_string_in.lstrip("#")
    if len(hex_string) == 6:
        argb_tuple = tuple(
            [1.0] + list(int(hex_string[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        )
    elif len(hex_string) == 8:
        argb_tuple = tuple(int(hex_string[i : i + 2], 16) / 255.0 for i in (0, 2, 4, 6))
    else:
        raise ValueError(
            "Color string must either be 6 or 8 chars long, current value: {}".format(
                hex_string
            )
        )
    rgba_dict = {
        "r": argb_tuple[1],
        "g": argb_tuple[2],
        "b": argb_tuple[3],
        "a": argb_tuple[0],
    }
    return rgba_dict


def generate_equally_spread_hues(
    nb_colors: int,
    saturation: float = 1.0,
    brightness: float = 1.0,
    transparency: float = 0.5,
):
    hsv_tuples = [
        (hue, saturation, brightness) for hue in generate_intervals_values(nb_colors)
    ]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    rgba_tuples = [(rgb[0], rgb[1], rgb[2], transparency) for rgb in rgb_tuples]
    return rgba_tuples


def generate_equally_spread_ros_colors(
    nb_colors: int,
    saturation: float = 1.0,
    brightness: float = 1.0,
    transparency: float = 0.5,
):
    return [
        ColorRGBA(r=r, g=g, b=b, a=a)
        for r, g, b, a in generate_equally_spread_hues(
            nb_colors, saturation, brightness, transparency
        )
    ]


def generate_intervals_values(nb_values: int) -> t.List[float]:
    if nb_values == 0:
        return []
    if nb_values == 1:
        return [0.0]
    if nb_values < 0:
        raise ValueError("nb_values must be positive.")
    else:
        intervals = [[0.0, 0.85]]
        values = [0.0, 0.85]
        while len(values) < nb_values:
            new_intervals = []
            for interval in intervals:
                middle = sum(interval) / 2
                new_intervals.append([interval[0], middle])
                new_intervals.append([middle, interval[1]])
                values.append(middle)

                if len(values) == nb_values:
                    break
            intervals = new_intervals

        return values


def blend_colors(colorRGBA1: ColorRGBA, colorRGBA2: ColorRGBA) -> ColorRGBA:
    alpha = 1.0 - ((1.0 - colorRGBA1.a) * (1.0 - colorRGBA2.a))
    red = colorRGBA1.r * (1.0 - colorRGBA2.a) + colorRGBA2.r * colorRGBA2.a
    green = colorRGBA1.g * (1.0 - colorRGBA2.a) + colorRGBA2.g * colorRGBA2.a
    blue = colorRGBA1.b * (1.0 - colorRGBA2.a) + colorRGBA2.b * colorRGBA2.a
    return ColorRGBA(r=red, g=green, b=blue, a=alpha)


def rgba_to_hex(r: float, g: float, b: float, a: float):
    return "#%02x%02x%02x%02x" % (
        int(a * 255),
        int(r * 255),
        int(g * 255),
        int(b * 255),
    )


def darken(hex_string: str, multiplier: float = 0.6):
    rgba_dict = hex_to_rgba(hex_string)
    hsv = colorsys.rgb_to_hsv(rgba_dict["r"], rgba_dict["g"], rgba_dict["b"])
    darker_v = min(1.0, max(0.0, hsv[2] * multiplier))
    darker_rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], darker_v)
    out_hex_string = rgba_to_hex(
        darker_rgb[0], darker_rgb[1], darker_rgb[2], rgba_dict["a"]
    )
    return out_hex_string


robot_color = ColorRGBA(**hex_to_rgba("#ff6d9eeb"))
movable_obstacle_color = ColorRGBA(**hex_to_rgba("#fff1c232"))
unmovable_obstacle_color = ColorRGBA(**hex_to_rgba("#ff000000"))
unknown_obstacle_color = ColorRGBA(**hex_to_rgba("#ff8e7cc3"))
robot_border_color = ColorRGBA(**hex_to_rgba("#ff1155cc"))
movable_obstacle_border_color = ColorRGBA(**hex_to_rgba("#ff7f6000"))
unmovable_obstacle_border_color = ColorRGBA(**hex_to_rgba("#ff4d2802"))
unknown_obstacle_border_color = ColorRGBA(**hex_to_rgba("#ff351c75"))
g_fov_border_color = ColorRGBA(**hex_to_rgba("#ff6d9eeb"))
s_fov_border_color = ColorRGBA(**hex_to_rgba("#ff6aa84f"))
min_inflated_polygon_border_color = ColorRGBA(**hex_to_rgba("#ff666666"))
max_inflated_polygon_border_color = ColorRGBA(**hex_to_rgba("#ff666666"))
text_color_on_filling = ColorRGBA(**hex_to_rgba("#ffffffff"))
text_color_on_empty = ColorRGBA(**hex_to_rgba("#ff000000"))
init_blocking_areas_color = ColorRGBA(**hex_to_rgba("#aafd5454"))
target_blocking_areas_color = ColorRGBA(**hex_to_rgba("#aac85ab7"))
init_diameter_inflated_polygon_color = ColorRGBA(**hex_to_rgba("#aa88dc7a"))
target_diameter_inflated_polygon_color = ColorRGBA(**hex_to_rgba("#aa24641a"))
transit_path_color = ColorRGBA(**hex_to_rgba("#ff6d9eeb"))
transfer_path_color = ColorRGBA(**hex_to_rgba("#ffe06666"))

r0_light_blue = ColorRGBA(**hex_to_rgba("#ff6d9eeb"))
r0_dark_blue = ColorRGBA(**hex_to_rgba("#ff3c78d8"))
r1_light_green = ColorRGBA(**hex_to_rgba("#ff93c47d"))
r1_dark_green = ColorRGBA(**hex_to_rgba("#ff6aa84f"))
r2_light_pink = ColorRGBA(**hex_to_rgba("#ff8e7cc3"))
r2_dark_pink = ColorRGBA(**hex_to_rgba("#ff674ea7"))
r3_light_red = ColorRGBA(**hex_to_rgba("#ffe06666"))
r3_dark_red = ColorRGBA(**hex_to_rgba("#ffcc0000"))

robot_color_r2 = ColorRGBA(**hex_to_rgba("#ffeb6ddd"))
robot_border_color_r2 = ColorRGBA(**hex_to_rgba("#ffcc11b2"))
transit_path_color_r2 = ColorRGBA(**hex_to_rgba("#ffeb6ddd"))
transfer_path_color_r2 = ColorRGBA(**hex_to_rgba("#ff6deb7b"))

flashy_green = ColorRGBA(**hex_to_rgba("#ff25ff00"))
flashy_cyan = ColorRGBA(**hex_to_rgba("#ff85ffff"))
flashy_purple = ColorRGBA(**hex_to_rgba("#ffff00ff"))
flashy_red = ColorRGBA(**hex_to_rgba("#ffff0000"))
flashy_dark_green = ColorRGBA(**hex_to_rgba("#ff007700"))
flashy_dark_cyan = ColorRGBA(**hex_to_rgba("#ff007777"))
dark_purple = ColorRGBA(**hex_to_rgba("#ff8e7cc3"))
dark_brown = ColorRGBA(**hex_to_rgba("#ffc3a87c"))
dark_blue = ColorRGBA(**hex_to_rgba("ff7ca5c3"))
black = ColorRGBA(**hex_to_rgba("#ff000000"))


if __name__ == "__main__":
    values_1 = generate_equally_spread_hues(1)
    values_5 = generate_equally_spread_hues(5)
    values_10 = generate_equally_spread_hues(10)
    print("")
