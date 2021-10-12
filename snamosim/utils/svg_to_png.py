import os

svg_names = {name for name in os.listdir("./") if ".svg" in name}

for svg_name in svg_names:
    os.system("inkscape " + svg_name + " -e " + svg_name + ".png")
