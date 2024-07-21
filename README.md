# Thomas Madeley Hiscox TF IDF 

### Context. 

Hi there, thank you for taking the time to read my submission. I will briefly summarise my approach and attempt to explain
some of the decision-making process and tradeoffs made. 


## Task


# Instructions


## Set up and installation

We have created a `Makefile` to house some helpful commands to manage dependencies and set up a suitable Conda environment. 

You must have some form of Conda installed for this to work. 

#### Create Environment and install the package in editable mode
Run the command below from the root directory, and answer `y` when asked about installation of dependencies.

This will automatically create a conda environment and ipykernel kernel for use in a jupyter notebook. 

```shell
make create_environment
```

If you need to update the dependencies, simply add the new package name(s) to `requirements.in` and then run:

```shell
make update_requirements

make dev_install
```

This will use pip-tools to search for compatible versions of all dependencies, then install the new dependencies in your conda environment. 


## TL;DR, Give Me A Brief Summary


## If I Had More Time
- Add more parameters to the functions
  - N Grams