# Training models

## how to train a model

1. Generate .npz files from the raw dataset. 


2. Run the training script with the desired parameters. The script will search for all runs in the dataset/data folder and train a model based on the provided .npz files.

```bash
cd ~/GIT/mapless/src/Navigation/mapless_navigation/mapless_nav/scripts/TorchModels

# E01
./train.py E01

# run all dataset
./train.py

# absolute path
./train.py /other/path/data/E01

# with plot
./train.py E01 --plot

# set epochs
./train.py E01 --epochs 50
```