import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="tactics2d",
    version="0.0.1",
    author="Yueyuan Li",
    author_email="rowena.academic@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX :: Linux",
    ]
)