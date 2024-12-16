# import os
# import gzip
# import pickle

# # import cedalion
# import cedalion.vis.plot_probe as vPlotProbe


# #rootDir_data = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Esplanade/'
# #rootDir_data = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Cocktail_party_whole_head_data/'
# #rootDir_data = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/BallSqueezing_WHHD/'
# rootDir_data = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/gradCPT_NN24a_pilot/'
# file_path_pkl = os.path.join(rootDir_data, 'derivatives', "blockaverage.pkl.gz")

# # load [blockaverage_all, rec.geo2d, rec.geo3d] from pickle file
# with gzip.GzipFile(file_path_pkl) as fin:        
#     blockaverage_all, geo2d, geo3d = pickle.load(fin)

# vPlotProbe.run_vis(snirfData = blockaverage_all, geo2d = geo2d, geo3d = geo3d)





import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog

import cedalion.vis.plot_probe as vPlotProbe

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
        blockaverage_all, geo2d, geo3d  = pickle.load(f)

    vPlotProbe.run_vis(snirfData = blockaverage_all, geo2d = geo2d, geo3d = geo3d)
else:
    print("No file selected.")