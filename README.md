# power_grids_ERG

In this repo there is the main implementation for the joint work with A. Zocca and G. Bet, "Generating Synthetic Power Grids using Exponential Random Graph Models". In "Tutorial_PG_ERG.ipynb" you will find a brief notebook with an example of the full generation procedure for a test grid (300_ieee). The default ERG model is the "edg_tri_2tri" which includes 6 edge-type counts, triangles count and 2-triangles count. Other implemented models are "edg" (6 edge-type count),  "edg_ddeg" (6 edge-type counts + each single node degree) and "edg_dgen" (6 edge-type counts + single node degree of each generator-type node).

# Description of the files:

- In pg_ergm_eest.py you will find an implementation of the main algorithms described in the paper. The EEsparse algorithm is the revised Equilibrium Expectation algorithm used to estimate the parameters of the ERG model in the constrained space of connected graphs, pg_MHergm_conn instead is the MH sampler to generate connected graphs. These algorithms are implemented in a way that fits in principle any ERG formulation provided that a function to compute the hamiltonian and one function to update the observables are provided.

-  pg_betas_comp.py contains functions dedicated to retrieve starting points for the parameters of various ERG models tested for power grids generation.

-  pg_ham_comp.py contains  functions to  compute and update observables of ERG models tested for power grids generation.

-  pg_utils contains various ancillary functions used in the code.

-  load_pglib_opf.py is a revisited version of the module created by Leon Lan (whom I still deeply thank for the help) available here https://github.com/leonlan/pglib-opf-pyparser.git , and it's used to parse the grid data.

