from setuptools import setup

setup(
    name="namedtensor",
    version="0.0.2",
    author="Alexander Rush",
    author_email="srush@seas.harvard.edu",
    packages=["namedtensor", "namedtensor.text", "namedtensor.nn",
              "namedtensor.distributions"],
    package_data={"namedtensor": []},
    url="https://github.com/harvardnlp/NamedTensor",
    install_requires=["torch", "torchtext", "numpy", "einops", "opt-einsum"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]

)
