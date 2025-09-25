# CSI: Conformalizing Statistical Inference with TRUST and TRUST++

This repository contains the implementation of TRUST and TRUST++, along with scripts for reproducing the figures and tables from the paper *Distribution-Free Calibration of Statistical Confidence Sets*.

## Overview

CSI provides tools for performing conformal statistical inference using TRUST and TRUST++. The package includes functionality for running experiments and evaluating statistical confidence sets in a distribution-free framework.

## Installation

To set up the CSI package and its dependencies:

1. **Navigate** to the directory containing `setup.py`.

2. **Activate Conda** in your terminal:
    ```bash
    source activate
    ```

3. **Create the Conda environment** with all required dependencies:
    ```bash
    conda env create -f CSI_env.yml
    ```

4. **Activate the new environment**:
    ```bash
    conda activate CSI_env
    ```

5. **Install the CSI package locally**:
    ```bash
    pip install .
    ```

## Running Experiments

Scripts for running real data experiments are provided in the `results/` directory. Example commands:

```bash
# Running for tractable models
# BFF experiments
python results/BFF_experiments.py

# e-value experiments
python results/e_value_experiments.py

# KS experiments
python results/KS_experiments.py

# LR experiments
python results/LR_experiments.py
```

