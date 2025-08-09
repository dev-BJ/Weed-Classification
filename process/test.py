import scipy.io as sio
import os
from pprint import pprint

base_path = os.path.abspath(os.getcwd())
# print(base_path)
# Load the .mat file
# mat_contents = sio.loadmat(f'{base_path}/dataset/train/imagelabels.mat')
# pprint(mat_contents['labels'][0][0:100])

# Access variables from the dictionary
# The keys of the dictionary correspond to the variable names in the .mat file
# data_variable = mat_contents['variable_name']
root, dirs, files = os.walk(f'{base_path}/dataset/train')
print(files)