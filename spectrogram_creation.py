import pandas as pd
from code_functions import DF_to_segmented_DF,print_df,plot_segments_and_trail,fit_records_to_frame
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import scipy

def curve_amplitude(df):
    '''
    Can be used to quantify the degree of a turn by returning the distance it is from a straight line
    from its start to end point. Works on segments.
    Returns the mean.
    '''
    if 'numpy' not in globals():
        import numpy as np
    x1, y1 = df.iloc[0]['position_long'], df.iloc[0]['position_lat']
    x2, y2 = df.iloc[-1]['position_long'], df.iloc[-1]['position_lat']
    if (x2-x1) != 0:
        m = (y2 - y1) / (x2 - x1)
    else:
        m = 0
    c = y1 - (m * x1)
    distance = []
    for i in range(len(df)):
        x,y = df.iloc[i]['position_long'], df.iloc[i]['position_lat']
        amplitude = (y - (m * x + c))
        if abs(amplitude) <= 10**(-4):
            amplitude=0
        distance.append(amplitude)
    return distance
def rotate_and_curve_amplitude(df):
    '''
    Rotates the input trail to align its overall direction of growth with the x-axis, and then calculates
    the degree of turning in the trail by returning the distance it is from a straight line from its start 
    to end point. Works on segments.
    Returns the mean.
    '''
    # Extract starting and ending coordinates
    x1, y1 = df.iloc[0]['position_long'], df.iloc[0]['position_lat']
    x2, y2 = df.iloc[-1]['position_long'], df.iloc[-1]['position_lat']

    # Calculate rotation angle
    if (x2-x1) != 0:
        theta = np.arctan((y2 - y1) / (x2 - x1))
    else:
        theta = 0
    # Apply rotation matrix to each point in the trail
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords = df[['position_long', 'position_lat']].to_numpy()
    coords_rot = np.dot(R, coords.T).T

    # Calculate distance from rotated trail to x-axis
    c = np.mean(coords_rot[:, 1])
    distance = []
    for i in range(len(df)):
        y = coords_rot[i, 1]
        amplitude = abs(y - c)
        # if amplitude <= 10**(-4):
        #     amplitude=0
        distance.append(amplitude)
    return distance,coords_rot
def freq_analysis(df):
    # if 'scipy' not in globals():
    #     import scipy
    if 'climb' not in df.columns:
        df['climb']=df['altitude'].diff()
        df['climb'].iloc[0]=0
    time = (df['timestamp'].diff()/ np.timedelta64(1, 's')).to_numpy()
    time[0]=0
    time=np.cumsum(time)

    position_lat = df['Vposition_lat'].to_numpy()
    position_long = df['Vposition_long'].to_numpy()

    position_lat_fft = scipy.fftpack.fft(position_lat)
    position_long_fft = scipy.fftpack.fft(position_long)
    position_freq = np.fft.fftfreq(len(position_lat), time[1]-time[0])

    signal1_magnitude = np.abs(position_long_fft / len(position_lat))
    position_long_fft_db = 20 * np.log10(signal1_magnitude)
    df['long_fft_db']=position_long_fft_db

    signal1_magnitude = np.abs(position_lat_fft / len(position_lat))
    position_lat_fft_db = 20 * np.log10(signal1_magnitude)
    df['lat_fft_db']=position_lat_fft_db
    df['position_freq'] = position_freq

    climb = df['climb'].to_numpy()
    climb_fft = scipy.fftpack.fft(climb)
    climb_freq = np.fft.fftfreq(len(climb), time[1]-time[0])
    signal2_magnitude = np.abs(climb_fft / len(climb))
    signal2_db = 20 * np.log10(signal2_magnitude)
    df['climb_freq'] = climb_freq
    df['climb_fft_db']=signal2_db
def spectrogram_correlation(mag1,mag2):
    if 'scipy' not in globals():
        import scipy as sc
    if mag1.shape==mag2.shape:
        correlation = sc.signal.correlate(mag1,mag2)
        return correlation
    else:
        window_size = mag1.shape[1]
        shift_size = mag2.shape[1] - window_size
        print('shift size',shift_size)
        for i in range(0,shift_size):
            mag2_window=mag2[:,i:i+window_size]
            print((i,i+window_size))
            print(mag2_window,'\n')
def split_freq_bands(spectrogram, freq_list, bottom_freq, top_freq, num_bins):
    """
    Split the frequency bands of a spectrogram into evenly spaced bins.

    Args:
        spectrogram (np.ndarray): Magnitude spectrogram matrix (size: len(freq_list) x len(time)).
        freq_list (np.ndarray): List of frequencies corresponding to rows of the spectrogram.
        bottom_freq (float): Bottom frequency value for the lowest bin.
        top_freq (float): Top frequency value for the highest bin.
        num_bins (int): Number of bins to split the frequency range into.

    Returns:
        freq_bins (np.ndarray): List of frequencies for each bin (size: num_bins).
        energy (np.ndarray): Energy in each frequency bin for each time step in the spectrogram 
            (size: num_bins x len(time)).
    """

    # Calculate the frequency range of each bin
    freq_range = np.linspace(bottom_freq, top_freq, num_bins+1)
    freq_bins = (freq_range[1:] + freq_range[:-1])/2

    # Find the indices of the frequency range of each bin in the spectrogram
    bin_indices = np.searchsorted(freq_list, freq_range)

    # Split the spectrogram into frequency bins
    energy = np.zeros((num_bins, spectrogram.shape[1]))
    for i in range(num_bins):
        bin_energy = np.sum(spectrogram[bin_indices[i]:bin_indices[i+1], :], axis=0)
        energy[i, :] = bin_energy

    return freq_bins, energy

TDF0 = DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'))
file = 'data/Evening_Run.fit'
# TDF0 = DF_to_segmented_DF(fit_records_to_frame(file)).iloc[200:400]
TDF1 = DF_to_segmented_DF(fit_records_to_frame(file)).iloc[200:600]
plot_segments_and_trail(TDF0,Show_Plot=True)

avrg_samplefreq = 1/(np.mean(TDF0['seconds']))
sample_rate = avrg_samplefreq

amps,coords_rot = rotate_and_curve_amplitude(TDF0)
signal = np.array(amps)
window_size = 60 #sum(TDF0['seconds'])/(len(np.unique(TDF0['segments'])))  # seconds
window_samples = int(window_size * sample_rate)
overlap = window_samples // 2
f, t, Sxx = spectrogram(signal, fs=sample_rate, window='hann', nperseg=window_samples, noverlap=overlap, scaling='spectrum')
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.pcolormesh(t, f, np.log10(Sxx), cmap='inferno')
ax3.set_ylim([0, 0.15])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Frequency (Hz)')
ax3.set_title('Spectrogram for first section')
plt.show()
# avrg_samplefreq1 = 1/np.mean(TDF1['seconds'])
# sample_rate1 = avrg_samplefreq1
# window_samples1 = int(window_size * sample_rate1)
# overlap1 = window_samples1 // 2
# amps1,coords_rot1 = rotate_and_curve_amplitude(TDF1)
# signal1 = np.array(amps1)
# f1, t1, Sxx1 = spectrogram(signal1, fs=sample_rate1, window='hann', nperseg=window_samples1, noverlap=overlap1, scaling='spectrum')
# fig1, ax1 = plt.subplots(figsize=(6, 4))

# ax1.pcolormesh(t1, f1, np.log10(Sxx1), cmap='inferno')
# ax1.set_ylim([0, 0.15])
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Frequency (Hz)')
# ax1.set_title('Spectrogram for second section')
# plt.show()

# print('Window samples 1',window_samples1)
# print('Window samples',window_samples)

start_freq=0
end_freq=0.135
num_bins=15

# bin_edge0,bin_energy0 = split_freq_bands(Sxx,f,start_freq,end_freq,num_bins)
# bin_edge1,bin_energy1 = split_freq_bands(Sxx1,f1,start_freq,end_freq,num_bins)
# temp = spectrogram_correlation(bin_energy0,bin_energy1)
# print(bin_energy0.shape,bin_energy1.shape)
# print(temp)
