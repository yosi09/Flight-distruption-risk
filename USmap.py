# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:28:06 2018

@author: Yossi
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

def plot_area(pos):
    x, y = m(pos[1], pos[0])
    size =5# (pos[2]/1000) ** 2 + 3
    m.plot(x, y, 'o', markersize=size, color='#444444', alpha=0.8)

def plot_map(westlimit=-128.9, southlimit=24.2, eastlimit=-64.2, northlimit=49):
    #Coordinate of the USA
    westlimit=max(-128.9,westlimit)
    eastlimit=min(-64.2,eastlimit)
    southlimit=max(24.2,southlimit)
    northlimit=min(49,northlimit)
    
    fig, ax = plt.subplots(figsize=(15,10))
    
    m = Basemap(resolution='l', # c, l, i, h, f or None
                projection='merc',
                lat_0=54.5, lon_0=-4.36,
                llcrnrlon=westlimit, llcrnrlat= southlimit, urcrnrlon=eastlimit, urcrnrlat=northlimit)
    
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
    m.drawcoastlines()
    
    m.readshapefile('./cb_2016_us_state_5m/cb_2016_us_state_5m','states',drawbounds=True)
    return m
if __name__ == "__main__":
    m=plot_map()
    airports=pd.read_csv('./data/airportsLoc.csv',index_col=0)
    airports.apply(plot_area,axis=1)
