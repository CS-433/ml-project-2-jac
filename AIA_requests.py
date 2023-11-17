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

"""
============================================================================================
1st line of data 

date: 2011-01-20T09:15:44.000000
basepoint_x: -226.577125	
basepoint_y: -956.964375

2011-01-20T23:49:20.000000
-143.625000
386.404000
============================================================================================
"""

start_time = Time('2011-01-20T09:15:44.000000', scale='utc', format='isot') 
bottom_left = SkyCoord(-426*u.arcsec, -1156*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")
top_right =SkyCoord(-26*u.arcsec, -756*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")

cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True)

#print(os.environ)
jsoc_email = "julie.charlet@epfl.ch"

query = Fido.search(
    a.Time(start_time - 10*u.min, start_time + 10*u.min),
    a.Wavelength(304*u.angstrom),
    a.Sample(2*u.min),
    a.jsoc.Series.aia_lev1_euv_12s,
    a.jsoc.Notify(jsoc_email),
    a.jsoc.Segment.image,
    cutout,
)
print(query)

files = Fido.fetch(query)
files.sort()

sequence = sunpy.map.Map(files, sequence=True)

fig = plt.figure()
ax = fig.add_subplot(projection=sequence.maps[0])
ani = sequence.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5e3, stretch=SqrtStretch()))

plt.show()
"""
============================================================================================
Importing more than one line at a time using functions
============================================================================================


data=load_data()
i = 0
row_data = data.iloc[i]

files = get_image(data) # get the images for the line i

sequence = sunpy.map.Map(files, sequence=True)

fig = plt.figure()
ax = fig.add_subplot(projection=sequence.maps[0])
ani = sequence.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5e3, stretch=SqrtStretch()))

plt.show()


for i in range(2):
    files = get_image(data[data.index==i]) # get the images for the line i
    image_array = array_file(files)

    sequence = sunpy.map.Map(files, sequence=True)

    sequence_array = sequence.as_array()

    print(sequence_array)
    fig = plt.figure()
    ax = fig.add_subplot(projection=sequence.maps[0])
    ani = sequence.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5e3, stretch=SqrtStretch()))

    plt.show()




files = get_image(data[data.index==0]) #first line of the data
image_array = array_file(files)

sequence = sunpy.map.Map(files, sequence=True)

fig = plt.figure()
ax = fig.add_subplot(projection=sequence.maps[0])
ani = sequence.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5e3, stretch=SqrtStretch()))

plt.show()

"""