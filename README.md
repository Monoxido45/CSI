# CSI: Conformalizing Statistical Inference with TRUST and TRUST++


Here we provide an implementation of TRUST and TRUST++ and scripts to reproduce the figures and tables from the paper Distribution-Free Calibration of Statistical Confidence Sets ([check paper here](https://arxiv.org/abs/2411.19368)).

## Installing Dependencies and Package

To install the necessary dependencies and the CSI package, follow these steps:

1. Navigate to the directory containing the `setup.py` file.

2. Activate conda in the terminal
    ```bash
    source activate
    ```
    
2. Install the local conda environment with all dependencies by running the following command:
    ```bash
    conda env create -f CSI_env.yml
    ```

3. Activate the conda environment:
    ```bash
    conda activate CSI_env
    ```

4. Install the CSI package:
    ```bash
    pip install .
    ```

## Running Real Data Experiments

To run all experiments for tractable experiments, use commands like:
```bash
# for BFF
python results/BFF_experiments.py

# for e-value
python results/e_value_experiments.py

# for KS
python results/KS_experiments.py

# for LR
python results/LR_experiments.py
```

