# Stable Port-Hamiltonian Neural Networks

This code base contains the code to reproduce the results from the `Stable Port-Hamiltonian Neural Networks` paper.

See `requirements.txt` for the required packages and Python version to run the code in each Jupyter notebook from the `experiments` directory.

The `cascaded tanks` and `additive manufacturing surrogate` experiments require data from external sources to be placed in the respective directories in `data/` as described in the corresponding notebooks.
The data pertaining to the remaining experiments is included within this codebase and has been provided with the explicit consent of the respective authors.

Trained model weights are included and are loaded per default in each script. To rerun any experiment change the `save_dir` variable in each notebook under the section "Set Hyperparameters" to the new directory where the weights from the rerun should be saved.
