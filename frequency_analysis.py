from code_functions import *
from scipy.fftpack import fft,fftfreq

def fourier_transform(magnitudes):
    # Convert the input list to a numpy array
    data = np.array(magnitudes)

    # Perform Fourier Transform on the data
    fourier = np.abs(fft(data))

    return fourier
def plot_single_sided_spectrum(frequencies, magnitudes):
    # Discard the negative frequencies
    frequencies = frequencies[:len(frequencies)//2]
    magnitudes = magnitudes[:len(magnitudes)//2]
    # Plot the spectrum
    plt.stem(frequencies, magnitudes)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()
def extract_frequencies_time(magnitudes, time):
    fourier = fourier_transform(magnitudes)
    # Number of samples in the original signal
    N = len(magnitudes)
    # Time duration of the signal
    T = np.cumsum(time)[len(time)-1]
    # Sample rate
    sample_rate = N / T
    # Extract the frequencies
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)
    return frequencies
# TDF0=DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'),Weird_format=True)
TDF0=DF_to_segmented_DF(pd.read_parquet('data/2022-06-05-12-12-09 (1).parquet'))#.iloc[:500])
# print(SDF0)
def frequency_analysis(TDF0,inplace=True):
    '''
    Does some Fourier analysis and returns the maximum value and its frequency.
    Works on RAW TRAILS, but returns SEGMENT DATAFRAME.
    '''
    if 'timestamp' not in TDF0.columns:
        pp_time = [0]*TDF0.shape[0]
        pp_distance=TDF0['distance'].diff()
        for i in range(0,TDF0.shape[0]):
            pp_time[i] = pp_distance[i]/TDF0['velocity [m/s]'].iloc[i]
        pp_time[0]=0
        timestamp = np.cumsum(pp_time)
        TDF0['pp_time'] = pp_time
        TDF0['timestamp']=timestamp
    SDF0=(segments_to_feature_df(TDF0))
    classify_feature_df(SDF0) 
    segments = TDF0['segments'].tolist()
    amps=[0]*(segments[-1]+1)
    freqs=[0]*(segments[-1]+1)
    # plot_segments_and_trail(TDF0,Show_Plot=True)
    for i in range(segments[-1]+1):
        segment = TDF0.loc[TDF0['segments']==i]
        signal,time = df_curve_to_signal_rep(segment)
        fourier = np.delete(fourier_transform(signal),0)
        frequencies = np.delete(extract_frequencies_time(signal, time),0)
        amps[i]= max(fourier)
        freqs[i] = frequencies[np.argmax(fourier)]
        # print(i,round(frequencies[np.argmax(fourier)]*10**3,3),SDF0['class'].iloc[i])
    if inplace:
        SDF0['freq_amplitude']=amps
        SDF0['freq']=freqs
    return SDF0
SDF0=frequency_analysis(TDF0)
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
for kind in np.unique(SDF0['class'].tolist()):
    points=SDF0.loc[SDF0['class']==kind]
    ax.scatter(points['freq'],points['freq_amplitude'],label=kind)
ax.axhline(0.002,c='r')
ax.legend()
ax.grid()
ax.set_xlabel('Frequency')
plt.show()