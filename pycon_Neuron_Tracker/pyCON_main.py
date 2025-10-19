# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:52:53 2022

@author: Nikhil
"""

file = 'sample_movie.tif' # movie filename

#min and max sigma values for Difference of Gaussian. This can affect edge detection.
#%%
gaussian_filt_min = 3
gaussian_filt_max = 20
sampling_int  = 60 # sampling interval in minutes
#%%

#Set plotting parameters
#%%
scatter_markersize = 12
maps_minper = 'auto'
maps_maxper  = 'auto'
amplitudemap_color = 'viridis_r'
periodmap_color = "jet" #colorscale for periodmaps
phasemap_color = "hsv" #colorscale for phasemaps (preferably use cyclic colormaps)
map_background_color = 'white'#background color for all plots mentioned above
clustermap_color = "Set1" #colormaps for clustermap
dpi = 300
#%%

#Scripts
#%%
exec(open("./Scripts/pycon_neuron_tracker.py").read())
# %%
