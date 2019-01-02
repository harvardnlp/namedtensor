from setuptools import setup

setup(
    name="namedtensor",
    version="0.0.1",
    author="Alexander Rush",
    author_email="srush@seas.harvard.edu",
    packages=["namedtensor"],
    package_data={"namedtensor": []},
    url="https://github.com/harvardnlp/NamedTensor",
    install_requires=["numpy", "einops"],
)
