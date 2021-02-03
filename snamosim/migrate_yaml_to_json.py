import fnmatch
import os
import yaml
import json


if __name__ == "__main__":
    working_directory = "/home/xia0ben/INRIA/Code/s-namo-sim/data/simulations/"

    matches = []
    for root, dirnames, filenames in os.walk(working_directory):
        for filename in fnmatch.filter(filenames, '*.json'):
            matches.append(os.path.join(root, filename))

    for match in matches:
        with open(match) as f:
            data = yaml.load(f, yaml.SafeLoader)

        if "files" in data:
            if "world_file" in data["files"]:
                data["files"]["world_file"] = data["files"]["world_file"].replace(".yaml", ".json")

        with open(match, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)
