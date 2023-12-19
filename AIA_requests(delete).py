""" Requesting cutouts of AIA images from the JSOC """

import os

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch

import sunpy.coordinates  # NOQA
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
from functions_AIA import *

import torch

#######################
#Working version on Adrien_messy.ipynb
#######################


# Load the csv data containing the events
data = load_data()
# Get the number of event and initialize the output array
#events_list = [data[data.index==i] for i in range(1,2)] # Will become the full list of event when we actually download everything
events_list = data.iloc[:1] # Will become the full list of event when we actually download everything
#N_events = data.shape[0] : for when we download everything
N_events = len(events_list)
num_images = 30
# data_array = np.zeros((N_events, 166, 166, num_images))   
# last dim is the number of images per sequence => needs to be consistent with the end_time and sample time in the get_image function

''' This could work if we don't want to delete files after each iteration
files = get_image(data.iloc[:5], num_images)
sequence_array = array_file(files)
np.savez_compressed('./data1/array.npz', sequence_array)
'''

list_arrays = []
for i in events_list.index:
    files = get_image_one_date(data.iloc[i], num_images)
    sequence_array = array_file(files)
    # Currently all the events would take around 8GB, so we have to do the downsampling_layer
    # I saw it was already implemented, but I had to comment it out because I was getting an error
    list_arrays.append(sequence_array)
    print('iteration ', i, ' done', sequence_array.shape)
    for j in range(len(files)):
        os.remove(files[j])     # Remove image file after saving it to the array

np.savez_compressed('./data1/array_list.npz', list_arrays)



'''
# This loop goes over each data line of the .csv file
for i, event in enumerate(events_list):
    files = get_image(event, num_images) # selects data line i
    # "files" is (class <parfive>) and contains N <HDUList> objects (where N is the nb of images in the sequence) 
    # that we open as "f". The attribute f.data returns a numpy array in our case, bc the data is an image.
    # sequence_array is a 3D array of shape (500, 500, N) which contains all the pixel values for one line of data (~ 1 event)
    sequence_array = array_file(files)
    print('Shape of sequence_array: ', sequence_array.shape)
    # Add the 3D array into the 4D array with all the data
    torch.save(sequence_array, "./data1/"+str(i)+".pt" )
    #data_array[i,:,:,:] = sequence_array

    # Plotting the numpy arrays to check that we get the images => need improvement to chech the image's quality
    """
    for j in range(len(files)):
        img = sequence_array[:,:,j]
        plt.imshow(img, cmap='hot', interpolation='nearest')
        plt.show()
    """
# Check the size of final array
#print(data_array.shape)

# Find a way to export the array to an external file the reuse in the model. Possibilities to look at:
# .npy format and np.save function
# HPF5 format (good with large amount of data)

#data_array32 = data_array.astype(np.float32)
#np.save("data_1.npy", data_array)

'''