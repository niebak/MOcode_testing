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
    print(theta)
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



# TDF0 = DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'))
file = 'data/Evening_Run.fit'
TDF0 = DF_to_segmented_DF(fit_records_to_frame(file)).iloc[200:400]
# plot_segments_and_trail(TDF0,Show_Plot=True)
# print_df(TDF0)
avrg_samplefreq = 1/(np.mean(TDF0['seconds']))
sample_rate = avrg_samplefreq
# print(sample_rate)
amps,coords_rot = rotate_and_curve_amplitude(TDF0)
signal = np.array(amps)
window_size = 60 #sum(TDF0['seconds'])/(len(np.unique(TDF0['segments'])))  # seconds
window_samples = int(window_size * sample_rate)
overlap = window_samples // 2
# print(window_size,window_samples,overlap,type(signal),sample_rate)
f, t, Sxx = spectrogram(signal, fs=sample_rate, window='hann', nperseg=window_samples, noverlap=overlap, scaling='spectrum')

# print(sum(TDF0['seconds']))

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.pcolormesh(t, f, np.log10(Sxx), cmap='inferno')
ax3.set_ylim([0, 0.5])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Frequency (Hz)')
ax3.set_title('Spectrogram')
# ax3.grid()

plt.show()

# freq_analysis(TDF0)
# freqfig=plt.figure()
# frexax1=freqfig.add_subplot(2,1,1)
# frexax2=freqfig.add_subplot(2,1,2)
# frexax1.plot(TDF0['position_freq'],TDF0['lat_fft_db'])
# frexax1.plot(TDF0['position_freq'],TDF0['long_fft_db'])
# frexax2.plot(TDF0['climb_freq'],TDF0['climb_fft_db'])
# plt.show()