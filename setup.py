from setuptools import setup, find_packages

REQUIRED = ["pyproj", "shapely", "gym"]

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
    name="tactics2d",
    version="0.0.1",
    author="Yueyuan Li",
    author_email="rowena.academic@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=REQUIRED,
    python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX :: Linux",
    ]
)