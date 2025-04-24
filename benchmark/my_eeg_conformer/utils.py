
import numpy as np
import matplotlib.pyplot as plt



def plot_eeg(eeg, label, channel_names=None):
    num_channels = eeg.shape[0] 
    num_points = eeg.shape[1]
    fig, axes = plt.subplots(num_channels, 1, sharex=True, figsize=(10, 2 * num_channels))
    for i in range(num_channels):
        axes[i].scatter(range(num_points), eeg[i, :], c=['red' if lbl else 'blue' for lbl in label], s=1)
        if channel_names:
            axes[i].set_ylabel(channel_names[i])
        else:
            axes[i].set_ylabel(f"Channel {i+1}")
        axes[i].grid(True)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    plt.show()



