
import h5py
import pandas as pd

# Load the HDF5 file as a DataFrame
with pd.HDFStore("/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/Agent/Experiments/history/Vatnsendi4_vld.h5", "r") as store:
    print(store.keys())  # List all stored datasets
    df = store["/Vatnsendi4"]  # Load the dataset
    print(df.head())  # Display the first few rows
