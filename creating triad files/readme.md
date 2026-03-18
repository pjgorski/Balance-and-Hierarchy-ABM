This folder contains folders specific to each dataset. Apart from additional dataset analysis, the purpose of Jupyter notebooks is to generate triad files that can be used in the agent-based model simulations. The proper format is described below. 

* The file can be of csv or hdf formats. 
* It should contain 6 columns, named: a, b, c, ab, ac, bc
* Each row describes one triad. 
* Columns a, b, c contain node IDs of P, O, X agents in ego-based triad, respectively. 
* Columns ab, ac, bc contain $\pm 1$ values describing signs of respective edges inside the triad. 

For each dataset a Jupyter notebook in Julia language is created. It is named `analyse_triads_julia_with_cycles.ipynb`. What is needed to run it:

* It was obtained for Julia 1.11.2. It requires two packages: CSV (version 0.10.15) and DataFrames (version 1.7.0). 
* One needs to download the dataset file. In the notebook, it is necessary to adjust file source. 
* The column names (a, b, c, ...) might need to be adjusted. 
* For high school dataset, one needs to save the files. 
