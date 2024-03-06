#!/bin/bash

# Download datasets
python3 data_scripts/download.py -d blog
python3 data_scripts/download.py -d protein
python3 data_scripts/download.py -d concrete
python3 data_scripts/download.py -d news
python3 data_scripts/download.py -d kernel
python3 data_scripts/download.py -d superconductivity
python3 data_scripts/download.py -d airfoil
python3 data_scripts/download.py -d electric
python3 data_scripts/download.py -d cycle
python3 data_scripts/download.py -d winered
python3 data_scripts/download.py -d winewhite

# Process datasets
python3 data_scripts/process.py -d blog
python3 data_scripts/process.py -d protein
python3 data_scripts/process.py -d concrete
python3 data_scripts/process.py -d news
python3 data_scripts/process.py -d kernel
python3 data_scripts/process.py -d superconductivity
python3 data_scripts/process.py -d airfoil
python3 data_scripts/process.py -d electric
python3 data_scripts/process.py -d cycle
python3 data_scripts/process.py -d winered
python3 data_scripts/process.py -d winewhite
python3 data_scripts/process.py -d bike
python3 data_scripts/process.py -d meps19
python3 data_scripts/process.py -d star
