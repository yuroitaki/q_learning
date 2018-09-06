## Set-Up

Install Anaconda, then run the following
command:

```
conda env create -f environment.yml
```

## Code Structure

All the scripts are located in the [tabular](src/tabular)
folder. There are four main components, whose details are written as
comments in each of their scripts respectively.

### Main driver [main.py](src/tabular/main.py)
This script is the main script used to conduct training and evaluation. A second main
driver, [monte_main.py](src/tabular/monte_main.py), is created for the Monte Carlo method.

### Agent class [t_agent.py](src/tabular/t_agent.py)
This script contains the implementation of our agent.

### Environment class [map_env.py](src/tabular/map_env.py) & [map_agent.py](src/tabular/map_agent.py)
This script contains the implementation of our environment. It has a
child class, which is a stochastic version of it [map_stoc_env.py](src/tabular/map_stoc_env.py).

### Helper function [helper.py](src/tabular/helper.py)
The helper function script that contains all the utility codes.

## Running the Code

The only script that should be used are the main drivers, unless
specific modifications are intended to be made to the agent or the environment.

To run the code, **cd** to the *src* folder, and run the following
command:

```
python -m tabular.main
```

or,

```
python -m tabular.monte_main
```

