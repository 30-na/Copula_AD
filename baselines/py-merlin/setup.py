#!/usr/bin/env python

# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: Apache-2.0

from distutils import util
from typing import Any

from setuptools import find_packages, setup  # type: ignore[import]

pkg_name = "merlin"
author = "Ferdinand Rewicki"
author_email = "ferdinand.rewicki@dlr.de"
description = (
    "A python implementation of the MERLIN anomaly detectino algorithm "
    'published by Nakamura et al. 2020 in "MERLIN: Parameter-Free '
    "Discovery of Arbitrary Length Anomalies in Massive Time Series "
    'Archives,"'
)
license = "Apache 2.0"
install_requires: list[str] = ["numpy~=1.21", " matplotlib~=3.4"]

##############################################
main_ns: dict[str, Any] = {}
ver_path = util.convert_path(f"{pkg_name}/_version.py")

with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name=pkg_name,
    version=main_ns["__version__"],
    python_requires=">=3.9",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[],
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pre-commit~=2.15",
            "pytest~=6.2",
            "pytest_mock~=3.7",
            "pytest-cov~=3.0",
            "mypy~=0.910",
        ],
        "docs": [
            "sphinx~=4.2",
            "sphinx-rtd-theme~=1.0",
            "sphinx-autodoc-typehints~=1.12",
        ],
    },
    author=author,
    author_email=author_email,
    description=description,
    license=license,
)
