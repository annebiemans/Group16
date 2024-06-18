from GitHub repositories, only use main (do not use branches). 
From main, first run data_filtering.ipynb, this uses tested_molecules.csv and creates the file descriptors.csv
Then run machinelearning.py, this uses the files descriptors.csv, tested_molecules.csv and untested_molecules.csv and rewrites untested_molecules.csv
