# import json
# import os
# from xml.dom import minidom
#
#
# def scenarios_from_simulation_results(scenario_original_filepath, scenario_logs_dir_filepath,
#                                       temp_simulations_dir_filepath, temp_worlds_dir_filepath):
#     # Get data from original files
#     with open(scenario_original_filepath) as f:
#         scenario_data = json.load(f)
#
#     world_file_path = os.path.join(os.path.dirname(scenario_original_filepath), scenario_data['files']['world_file'])
#     with open(world_file_path) as f:
#         world_data = json.load(f)
#
#     geometry_file_path = os.path.join(os.path.dirname(world_file_path), world_data['files']['geometry_file'])
#     with open(geometry_file_path) as f:
#         geometry_data = minidom.parse(f)
#
#     logged_scenarios_ids = {
#         name for name in os.listdir(scenario_logs_dir_filepath)
#         if os.path.isdir(os.path.join(scenario_logs_dir_filepath, name))
#     }
#
#     for scenario_id in logged_scenarios_ids:
#         sim_results_path = os.path.join(scenario_logs_dir_filepath, scenario_id, 'sim_results.json')
#
#         simulation_filepath = os.path.join(temp_simulations_dir_filepath, scenario_id + '/', os.path.basename(scenario_original_filepath))
#         world_json_filepath = os.path.join(temp_worlds_dir_filepath, scenario_id + '/', os.path.basename(world_file_path))
#         world_svg_filepath = os.path.join(temp_worlds_dir_filepath, scenario_id + '/', os.path.basename(geometry_file_path))
#
#         try:
#             with open(sim_results_path) as f:
#                 sim_results_data = json.load(f)
#
#             for agent_data in sim_results_data['agents']:
#                 agent_index = None
#                 for agent_counter, behavior_data in enumerate(scenario_data['agents_behaviors']):
#                     if behavior_data['agent_name'] == agent_data['agent_name']:
#                         agent_index = agent_counter
#
#                 if agent_index is None:
#                     continue
#
#                 if 'randomization' in scenario_data['agents_behaviors'][agent_index]['behavior']:
#                     del scenario_data['agents_behaviors'][agent_index]['behavior']['randomization']
#                 scenario_data['agents_behaviors'][agent_index]['behavior']['navigation_goals'] = []
#
#                 world_data['things']['zones']['goals'] = []
#
#                 for counter, goal_report in enumerate(agent_data['goals_reports']):
#                     goal_pose = goal_report['goal']
#                     goal_name = 'goal_' + str(counter)
#
#                     world_data['things']['zones']['goals'].append(
#                         {'name': goal_name, 'pose': goal_pose}
#                     )
#
#                     scenario_data['agents_behaviors'][agent_index]['behavior']['navigation_goals'].append({'name': goal_name})
#
#             # TODO Udpate filepath data for world svg in world json, and world json in simulation json
#             scenario_data['files']['world_file'] = os.path.join(
#                 os.path.relpath(os.path.dirname(world_json_filepath), os.path.dirname(simulation_filepath)),
#                 os.path.basename(world_json_filepath)
#             )
#             world_data['files']['geometry_file'] = os.path.join(
#                 os.path.relpath(os.path.dirname(world_svg_filepath), os.path.dirname(world_json_filepath)),
#                 os.path.basename(world_svg_filepath)
#             )
#
#             if not os.path.exists(os.path.dirname(simulation_filepath)):
#                 os.makedirs(os.path.dirname(simulation_filepath))
#             if not os.path.exists(os.path.dirname(world_json_filepath)):
#                 os.makedirs(os.path.dirname(world_json_filepath))
#
#             with open(simulation_filepath, 'w') as f:
#                 json.dump(scenario_data, f)
#             with open(world_json_filepath, 'w') as f:
#                 json.dump(world_data, f)
#             with open(world_svg_filepath, 'w') as f:
#                 geometry_data.writexml(f)
#         except (IOError, ValueError) as e:
#             continue
#
#
# def generate_after_the_feast_scenarios_from_simulation_results():
#     scenarios_from_simulation_results(
#         scenario_original_filepath=os.path.join(
#             os.path.dirname(__file__),
#             '../data/simulations/namo-socials/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset.json'
#         ),
#         scenario_logs_dir_filepath=os.path.join(
#             os.path.dirname(__file__),
#             '../logs/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset/'
#         ),
#         temp_simulations_dir_filepath=os.path.join(
#             os.path.dirname(__file__),
#             '../tmp/simulations/namo-socials/04_after_the_feast/variations-stilman_2005_behavior_complexified_random_goal_no_reset/'
#         ),
#         temp_worlds_dir_filepath=os.path.join(
#             os.path.dirname(__file__),
#             '../tmp/worlds/namo-socials/04_after_the_feast/variations-stilman_2005_behavior_complexified_random_goal_no_reset/'
#         )
#     )
#     scenarios_from_simulation_results(
#         scenario_original_filepath=os.path.join(
#             os.path.dirname(__file__),
#             '../data/simulations/namo-socials/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset_snamo.json'
#         ),
#         scenario_logs_dir_filepath=os.path.join(
#             os.path.dirname(__file__),
#             '../logs/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset_snamo/'
#         ),
#         temp_simulations_dir_filepath=os.path.join(
#             os.path.dirname(__file__),
#             '../tmp/simulations/namo-socials/04_after_the_feast/variations-stilman_2005_behavior_complexified_random_goal_no_reset_snamo/'
#         ),
#         temp_worlds_dir_filepath=os.path.join(
#             os.path.dirname(__file__),
#             '../tmp/worlds/namo-socials/04_after_the_feast/variations-stilman_2005_behavior_complexified_random_goal_no_reset_snamo/'
#         )
#     )

import base64
from io import BytesIO
from PIL import Image

# from PIL import ImageChops
import numpy as np
from xml.dom import minidom
import utils.conversion as conversion

if __name__ == "__main__":
    empty_svg = minidom.parse(
        "/home/xia0ben/INRIA/Code/s-namo-sim/namosim/utils/empty.svg"
    )
    conversion.set_all_id_attributes_as_ids(empty_svg)
    im = Image.open(
        "/home/xia0ben/INRIA/Code/s-namo-sim/data/thirdparties/gridsearch-dataset/www.movingai.com/benchmarks/dao/arena.png"
    )
    print(im.format, im.size, im.mode)
    # print(im[0][0])
    d_width, d_height = 49, 49
    out = im.resize((d_width, d_height))
    a = np.asarray(out)
    resolution = 0.1  # [m] Side length of a cell in meters
    width, height = d_width * resolution, d_height * resolution

    main_layer = empty_svg.getElementById("layer1")

    grid_layer = empty_svg.createElement("svg:g")
    grid_layer.setAttribute("id", "svg_grid_layer")
    grid_layer.setAttribute("inkscape:groupmode", "layer")
    grid_layer.setAttribute("inkscape:label", "Svg Grid Layer")
    main_layer.appendChild(grid_layer)

    svg_grid = empty_svg.createElement("svg:g")
    svg_grid.setAttribute("id", "svg_grid")
    grid_layer.appendChild(svg_grid)

    # b = np.fliplr(np.rot90(a, 3))
    b = np.flipud(np.rot90(a))
    for i in range(b.shape[0]):
        svg_grid_line = empty_svg.createElement("svg:g")
        svg_grid_line.setAttribute("id", "line_{}".format(i))
        svg_grid_line.setAttribute("line", str(i))
        svg_grid.appendChild(svg_grid_line)
        for j in range(b.shape[1]):
            svg_cell_path = conversion.rect2pathd(
                {
                    "x": i * resolution,
                    "y": j * resolution,
                    "width": resolution,
                    "height": resolution,
                }
            )
            svg_cell = empty_svg.createElement("svg:path")
            svg_cell.setAttribute("id", "svg_grid_cell_{}_{}".format(i, j))
            svg_cell.setAttribute("d", svg_cell_path)
            svg_cell.setAttribute(
                "style",
                "fill:{};fill-opacity:{}".format(
                    conversion.rgb_tuple_to_hex(b[i][j]), b[i][j][3] / 255.0
                ),
            )
            svg_cell.setAttribute("column", str(j))
            svg_grid_line.appendChild(svg_cell)

    out2 = Image.fromarray(a)

    buffered = BytesIO()
    out2.save(buffered, "PNG")
    b64str = base64.b64encode(buffered.getvalue())
    svg_image_str = "data:image/png;base64," + b64str
    svg_image = empty_svg.createElement("svg:image")
    svg_image.setAttribute("id", "grid")
    svg_image.setAttribute("xlink:href", svg_image_str)
    svg_image.setAttribute("style", "image-rendering:optimizeSpeed")
    svg_image.setAttribute("preserveAspectRatio", "none")
    svg_image.setAttribute("x", "0")
    svg_image.setAttribute("y", "0")
    svg_image.setAttribute("width", str(width))
    svg_image.setAttribute("height", str(height))

    main_layer.appendChild(svg_image)

    svg_root = empty_svg.getElementsByTagName("svg")[0]
    svg_root.setAttribute("width", str(width) + "cm")
    svg_root.setAttribute("height", str(height) + "cm")
    svg_root.setAttribute("viewBox", "0 0 {} {}".format(str(width), str(height)))

    with open(
        "/home/xia0ben/INRIA/Code/s-namo-sim/data/thirdparties/gridsearch-dataset/www.movingai.com/benchmarks/dao/arena-mod.svg",
        "w+",
    ) as f:
        empty_svg.writexml(f)
    # diff = ImageChops.difference(out, out2)
    # bbox = diff.getbbox()
    # if diff.getbbox():
    #     print('images are different')
    # else:
    #     print('images are the same')
    # im.show()
    out2.save(
        "/home/xia0ben/INRIA/Code/s-namo-sim/data/thirdparties/gridsearch-dataset/www.movingai.com/benchmarks/dao/arena-mod.png"
    )
