import os
import numpy as np


def load_data_for_DL(path):
    """
    Input:
    Path : strings to the folder path (./data0/ or ./data1/)

    Output:
    data : numpy array of shape [883,166,166,30] indicating : [number events, Height, Width, number images per event]
    """

    data=np.zeros((883,166,166,30))
    for i,file in enumerate(os.listdir(path)):
        archive=np.load(path+file)
        array=archive["arr_0"]
        data[i,:,:,:]=array

    return data