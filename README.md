# Balance and Hierarchy ABM
Authors: Adam Sulik, Piotr J. Górski

## Setup

1. At first make sure all below is installed:
* Python 3.8
    
    If it is not installed, and there is a more recent Python version, run the following code:

        ```
        sudo apt update
        sudo add-apt-repository ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install python3.8
        ```

* python-pip
* python-venv
    
Installing pip and venv for python3.8:

    ```
    sudo apt install python3.8-distutils python3.8-venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8
    ```

2. Then setup virtual environment writing in a terminal

    ```python3.8 -m venv /path/to/new/virtual/environment```.

    Suggested venv name is _\*venv\*_ eg. _cst_venv_. 
    
    Example: ```python3.8 -m venv _cst_venv_```

3. When it's done open your venv using command `source path/to/my/venv/bin/activate` (example: `source _cst_venv_/bin/activate`) and install required libraries using _requirements2.txt_ file:

    `pip install -r requirements2.txt`. Remember to open venv every time you start to work.

4. Check if your virtual environment is working properly by running example experiment: `python3.8 run-par.py`.

    It should run a number of small and short simulations with the output saved in the folder `outputs/LTDStatus/outputs/test-runs`. 

5. Check help for the CST `python3 main.py -h`

## Creating network files

The folder `creating triad files` contains description how this can be done. Note that to use provided code, you will need Julia or transfer the code to Python (for large datasets, it may be slow). 

## Running simulations

Files named `run-par-XXX.py` show how we used our ABM to generate proper simulations. See `run-par-2.py`, or simpler `run-par.py` for description or other files. 

## Analyzing simulations

The folder `experiments/phase_transition_exps` contain `*.py` files that help to analyze the simulations. They can obtain quasi-stationary level reached in a simulation, and then group results from different repetitions. This can be done in the remote server automatically, see files named as `*_script.py`. 

Jupyter notebooks show how we used the result outputs for specific cases for different datasets or for the complete graph. 
See readme in this folder. 

