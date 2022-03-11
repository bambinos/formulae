import codecs
import os

from setuptools import find_packages, setup

__author__ = "Tomás Capretto"
__author_email__ = "tomicapretto@gmail.com"

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
VERSION_FILE = os.path.join(PROJECT_ROOT, "formulae", "version.py")


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_version():
    with open(VERSION_FILE, encoding="utf-8") as buff:
        exec(buff.read()) # pylint: disable=exec-used
    return vars()["__version__"]


setup(
    name="formulae",
    version=get_version(),
    description="Formulas for mixed-effects models in Python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/bambinos/formulae",
    install_requires=get_requirements(),
    maintainer=__author__,
    maintainer_email=__author_email__,
    packages=find_packages(exclude=["tests", "test_*"]),
    license="MIT",
)
