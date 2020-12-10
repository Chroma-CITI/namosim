import colorsys
from std_msgs.msg import ColorRGBA


def hex_to_rgba(hex_string):
    hex_string = hex_string.lstrip('#')
    argb_tuple = tuple(int(hex_string[i:i + 2], 16) / 255. for i in (0, 2, 4, 6))
    rgba_tuple = (argb_tuple[1], argb_tuple[2], argb_tuple[3], argb_tuple[0])
    return rgba_tuple


def generate_equally_spread_hues(nb_colors, saturation=1., brightness=1., transparency=0.5):
    hsv_tuples = [(hue, saturation, brightness) for hue in generate_intervals_values(nb_colors)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    rgba_tuples = [(rgb[0], rgb[1], rgb[2], transparency) for rgb in rgb_tuples]
    return rgba_tuples


def generate_equally_spread_ros_colors(nb_colors, saturation=1., brightness=1., transparency=0.5):
    return [
        ColorRGBA(*color_tuple)
        for color_tuple in generate_equally_spread_hues(nb_colors, saturation, brightness, transparency)
    ]


def generate_intervals_values(nb_values):
    if nb_values == 0:
        return []
    elif nb_values == 1:
        return [0.]
    elif nb_values < 0:
        raise ValueError('nb_values must be positive.')
    else:
        intervals = [[0., 0.85]]
        values = [0., 0.85]
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


def blend_colors(colorRGBA1, colorRGBA2):
    alpha = 1. - ((1. - colorRGBA1.a) * (1. - colorRGBA2.a))
    red = (colorRGBA1.r * (1. - colorRGBA2.a) + colorRGBA2.r * colorRGBA2.a)
    green = (colorRGBA1.g * (1. - colorRGBA2.a) + colorRGBA2.g * colorRGBA2.a)
    blue = (colorRGBA1.b * (1. - colorRGBA2.a) + colorRGBA2.b * colorRGBA2.a)
    return ColorRGBA(red, green, blue, alpha)


if __name__ == '__main__':
    values_1 = generate_equally_spread_hues(1)
    values_5 = generate_equally_spread_hues(5)
    values_10 = generate_equally_spread_hues(10)
    print('')