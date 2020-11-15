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
)