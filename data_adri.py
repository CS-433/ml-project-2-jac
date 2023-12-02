import pandas as pd
import numpy as np
import os

from functions_AIA import *



# Animations
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from astropy.visualization import ImageNormalize, SqrtStretch

data1=load_data()
num_images=30
jsoc_email="adrien.joliat@epfl.ch"

events_list = data1.iloc[203:]
a=0

for i in events_list.index:
    a+=1
    files = get_images(data1.iloc[i], num_images, jsoc_email) # selects data line i
    # "files" is (class <parfive>) and contains N <HDUList> objects (where N is the nb of images in the sequence) 
    # that we open as "f". The attribute f.data returns a numpy array in our case, bc the data is an image.
    # sequence_array is a 3D array of shape (166, 166, N) which contains all the pixel values for one line of data (~ 1 event)
    sequence_array = array_file(files) #torch array of the whole sequency event

    #Save the array
    np.savez_compressed("./data1/"+str(i)+".npz" , sequence_array)
    
    #Remove images
    for j in range(len(files)):
        os.remove(files[j])

    # Check the size of final array
    print(f"Download = {a}/{len(events_list.index)} and array ={sequence_array.shape}")