# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: CC0-1.0

install:
	pip install . --upgrade 

dev:
	pip install -e .[dev] --upgrade 
	pre-commit install

test:
	pytest

types:
	mypy merlin

coverage:
	pytest --cov-report html --cov merlin tests/

docu:
	pip install .[docs] --upgrade
	sphinx-build -b html docs/source/ docs/build/html

get_version:
	@printf "v"
	@python setup.py --version
