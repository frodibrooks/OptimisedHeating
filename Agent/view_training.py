
import h5py
import pandas as pd

# Load the HDF5 file as a DataFrame
with pd.HDFStore("/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/Agent/Experiments/test1.h5", "r") as store:
    print(store.keys())  # List all stored datasets
    df = store["scenes"]  # Load the dataset
    print(df.head())  # Display the first few rows
