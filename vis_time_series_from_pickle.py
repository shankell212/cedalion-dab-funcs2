# import os
# import gzip
# import pickle

# import cedalion.vis.time_series as vTS

# # ask user to select a file


# #rootDir_data = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Esplanade/derivatives/processed_data'
# #rootDir_data = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/gradCPT_NN24a_pilot/derivatives/processed_data'
# rootDir_data = '/Users/dboas/Downloads/avl/derivatives/processed_data'
# with gzip.open( os.path.join(rootDir_data, 'rec.pkl'), 'rb') as f:
#      rec = pickle.load(f)

# vTS.run_vis( rec )


import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog

import cedalion.vis.time_series as vTS

# Create a Tkinter root window (it will not be shown)
root = tk.Tk()
root.withdraw()

# Open a file dialog to select a file
file_path = filedialog.askopenfilename(
    initialdir='/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets',
    title='Select a data file',
    filetypes=(('gZip files', '*.gz'), ('Pickle files', '*.pkl'), ('All files', '*.*'))
)

# Check if a file was selected
if file_path:
    with gzip.open(file_path, 'rb') as f:
        rec = pickle.load(f)

    vTS.run_vis(rec)
else:
    print("No file selected.")