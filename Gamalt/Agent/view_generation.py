import h5py
import pandas as pd

# Load the HDF5 file
file_path = r"\Users\frodi\Documents\OptimisedHeating\Agent\Experiments\test2.h5"

# Open the HDF5 file
with h5py.File(file_path, 'r') as f:
    # Print all keys (datasets or groups)
    print("Keys in HDF5 file:")
    for key in f.keys():
        print(key)

    # If you want to read a specific dataset as a DataFrame, you can use:
    # df = pd.read_hdf(file_path, key="your_key_here")
    # But for now, let's just load everything and print its shape
    df = pd.read_hdf(file_path)
    print(df.shape)

print("Code has ran")
