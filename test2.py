
# IMPORTS
import importlib
import numpy as np
import utils
from os import listdir
from os.path import isfile, join
importlib.reload(utils)



boxnames = utils.get_delta_T_boxes(mypath='C:\Outputs\Outputs')

print(boxnames)


# %%

slices = []
for boxname in boxnames:

    box = utils.read_box(boxname, mypath='C:\\Outputs\\Outputs')

    for i in range(0,255,2):
        slice = box[i,:,:]
        slices.append(slice)
    for i in range(0,255,2):
        slice = box[i,:,:]
        slices.append(slice)
    for i in range(0,255,2):
        slice = box[i,:,:]
        slices.append(slice)




#C:\Outputs\Outputs
