import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch
from astropy.io import fits

import sunpy.coordinates  # NOQA
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

import torch


def load_data():
    data=pd.read_csv("./data/Jet_clusters_3.0_2.0_paperID.csv", sep=",")
    data.columns=data.columns.str.strip()
    data=data.drop(columns="velocity")

    return data

def get_image(data):
    
    dates=data["date"]
    
    for i,date in enumerate(dates):

        start_time = Time(date, scale='utc', format='isot') #“CCYY-MM-DDThh:mm:ss[.sss. . . ]”, Coordinated Universal Time (UTC), 

        bottom_x=data["basepoint_X"][i]  # I am not sure this is right
        bottom_y=data["basepoint_Y"][i]   # same

        duration=data["duration"][i]

        bottom_left = SkyCoord((bottom_x-200)*u.arcsec, (bottom_y-200)*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")
        top_right = SkyCoord((bottom_x+200)*u.arcsec, (bottom_y+200)*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")

        jsoc_email = "julie.charlet@epfl.ch"

        cutout = a.jsoc.Cutout(bottom_left=bottom_left, top_right=top_right, tracking=True)

        print(cutout)

        query = Fido.search(
            a.Time(start_time , start_time + duration*u.min), #duration is in min
            a.Wavelength(304*u.angstrom),
            a.Sample(2*u.min), #one image /12 s --> 5images per min
            a.jsoc.Series.aia_lev1_euv_12s,
            a.jsoc.Notify(jsoc_email),
            a.jsoc.Segment.image,
            cutout,
        )
        files = Fido.fetch(query)
        files.sort()
        
    return files


def array_file(file): #for the moment only designed for one file
    with fits.open(file[0]) as f:
        array=f[1].data
        image_array=np.array(array)
    return image_array


def plot(files):
    sequence = sunpy.map.Map(files, sequence=True)

    fig = plt.figure()
    ax = fig.add_subplot(projection=sequence.maps[0])
    ani = sequence.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5e3, stretch=SqrtStretch()))
    return ani