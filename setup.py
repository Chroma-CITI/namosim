import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="s-namo-sim",
    version="0.3.0",
    author="Benoit RENAULT",
    author_email="xia0ben-contact-pypi@littleroot.net",
    description="NAMO and S-NAMO Algorithms and tools for robotics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/brenault/s-namo-sim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
    install_requires=[
        # CORE LIBRARIES (used about everywhere)
        'numpy', # For all matrix-based logic
        'shapely', # For all geometry-based logic
        'matplotlib', # For standalone display without ROS Rviz, and for debug purposes
        'aabbtree', # Used to accelerate collision check detection
        # LOOSELY COUPLED LIBRARIES (used just in a few instances, could be easily replaced)
        'mapbox_earcut', # To triangulate polygons (required for conversion to ROS messages)
        'bidict', # Two-way dictionnary library used in simulator main class
        'svgpath2mpl', # Used to convert SVG geometries to Shapely geometries
        'Pillow', # Used to convert grids to images
        'scikit-image',  # Used to rasterize polygon contours and for image skeleton computation
        'scipy' # Used for skeleton computation
    ],
)
