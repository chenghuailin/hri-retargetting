from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os


def low_pass_filter(data, cutoff_freq=0.2, order=4):
    """
    Apply a low-pass Butterworth filter to the data.
    
    Parameters:
        data (numpy.ndarray): The input data to be filtered.
        cutoff_freq (float): The normalized cutoff frequency (0~1).
        order (int): The order of the Butterworth filter.
        
    Returns:
        numpy.ndarray: The filtered data.
    """
    b, a = signal.butter(order, cutoff_freq, 'low')
    filtered_data = data.copy()
    for idx in range(filtered_data.shape[1]):
        filtered_data[:, idx] = signal.filtfilt(b, a, data[:, idx])
    return filtered_data

def plot_qpose(data, dof_names):
    """
    Plot the qpose data.
    
    Parameters:
        data (numpy.ndarray): The filtered data to be plotted.
        dof_names (list): The names of the degrees of freedom (DOF).
    """
    x = np.arange(data.shape[0])  # or use actual x-coordinates, like time series

    # Plot dof lines
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):  # Iterate through each column
        plt.plot(x, data[:, i], label=f'{dof_names[i]}')  # Add legend with label

    # Add labels and title
    plt.xlabel('X-axis (e.g., Time)')
    plt.ylabel('Y-axis (e.g., Value)')
    plt.title('QPose')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside
    plt.grid(True)
    plt.tight_layout()  # Prevent legend from being obscured
    plt.show()


