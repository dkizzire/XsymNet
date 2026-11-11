import numpy as np
import pandas as pd 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['keymap.pan'] = ['d']
mpl.rcParams['keymap.zoom'] = ['z']

def get_peak_count(path_to_xy_file, prominence=1.5, delim=None, skip_rows=None, plot_peaks=False, return_peak_indicies=False):
    '''
    Parameters
    ----------
    path_to_xy_file: str
        Path to xy file.
    prominence : float
        Required prominence of peaks
    delim : str
        delimiter for pd.read_csv 
        ',' : comma separated  
        '\s+' : multi space separated list
    skip_rows :int
        Number of rows to skip from top of file to get to data
    plot_peaks : bool
        If True : plots the xy data with vertical lines at peak positions
    teturn_peak_indicies : bool
        If True : returns a list of indices of peaks

    Returns
    -------
    npeaks : int
        number of peaks in xy file
    peaks[0] : Numpy array
        optional : index of each peak
    '''
    
    headers = ['x', 'y']
    temp_array = pd.read_csv(path_to_xy_file, delimiter=delim, names=headers, header=None, skiprows=skip_rows, usecols=['y'])
    # extract y column into list to identify peaks
    ylist = temp_array['y'].values.tolist()
    # call find_peaks
    peaks = find_peaks(ylist, prominence=prominence)
    # get number of peaks
    npeaks = len(peaks[0])

    #print("Peaks position:", peaks[0])
    #print('Number of peaks:', len(peaks[0]))

    if plot_peaks == True:
        # Plotting
        plt.plot(ylist)
        plt.title("Finding Peaks")

        [plt.axvline(p, c='C3', linewidth=0.3) for p in peaks[0]]

        plt.show()

    if return_peak_indicies == True:
        return npeaks, peaks[0]
    
    if return_peak_indicies == False:
        return npeaks