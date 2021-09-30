#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.1.2",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Juan Carlos Calvo Jackson",
    author_email="juancarlos.calvo@quantumblack.com",
    python_requires="==3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
    ],
    description="Udacity's first project basic implementation from scratch",
    entry_points={"console_scripts": ["navigate=navigate.cli:main",],},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="navigate",
    name="navigate",
    packages=find_packages(include=["navigate", "navigate.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/juan-carlos-calvo/navigate",
    version="0.1.0",
    zip_safe=False,
)