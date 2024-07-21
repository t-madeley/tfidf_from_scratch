
SHELL := /usr/bin/env bash

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME := hiscox_tfidf
PYTHON_INTERPRETER := $(shell which python)
VERSION := $(shell $(PYTHON_INTERPRETER) setup.py --version)
PIP_TOOLS_VERSION := 7.3.0

ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif

## Check the project's conda environment is active
_conda_env_check:
ifeq ($(CONDA_DEFAULT_ENV), base)
	@echo "You are in your base environment, please use the project dedicated environment: conda activate $(PROJECT_NAME)"
	@exit 1
endif

## Install/update pip tools for dependencies management
_install_pip_tools:
	$(PYTHON_INTERPRETER) -m pip install -U pip-tools=="$(PIP_TOOLS_VERSION)"

## Update requirements.txt with the required packages (from requirements.in)
update_requirements: _install_pip_tools
	sort --unique requirements.in --output requirements.in
	pip-compile --no-emit-index-url requirements.in


## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name $(PROJECT_NAME) python="3.11" -y --force
	@echo -n "Would you like to install the package and its dependencies? [y/N] " && read ans && [ $${ans:-N} = y ]
	@echo ">>> Installing the package and adding the kernel to Jupyter:"
	@zsh develop_install.sh
else
	@echo "Please install conda to create an environment"
endif
	@echo ">>> New env created and kernel installed within Jupyter. Activate with:"
	@echo "conda activate $(PROJECT_NAME)"

## Install the package in editable mode
dev_install: _conda_env_check _install_pip_tools
	pip-sync requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Lint using black
lint:
	black $(PROJECT_NAME)
	black tests
	isort --profile black $(PROJECT_NAME)
	isort --profile black tests
	pydocstyle $(PROJECT_NAME) -e --convention=numpy