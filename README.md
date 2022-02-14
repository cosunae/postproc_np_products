# postproc_np_products
numpy based version of some postprocessing products

## Installation

```conda env create -f environment.yml```

## Jupyter lab

In order to launch a jupyter notebook from CSCS, 
choose a <port> number and execute the following from a login node at CSCS

```
source setup.sh
conda activate postproc_np_products
jupyter lab notebook/brn_simple.ipynb --port <port>
```

And forward the chosen port (from your local host). 

```
ssh -N -f -L 127.0.0.1:<port>:127.0.0.1:<port> <user>@<tsa-login-node>.cscs.ch
```
Notice that <tsa-login-node>.cscs.ch must correspond to the actual host of the login node where jupyter is running (not an alias)


Now, you can open the jupyter lab in your browser at the following address: 

```
localhost:<port>
```
