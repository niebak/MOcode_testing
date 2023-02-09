import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from code_functions import DF_to_segmented_DF,plot_segments_and_trail
# df =DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'))
TDF0 = DF_to_segmented_DF(pd.read_parquet('data/2022-06-05-12-12-09 (1).parquet')).iloc[0:1200]
# time = df['distance']/df['speed'].mean()
# time[0]=0
segments = range(0,TDF0['segments'].iloc[-1])
for i in segments:
    df = TDF0.loc[TDF0['segments']==i].copy(deep=True)
    df['climb']=df['altitude'].diff()
    df['climb'].iloc[0]=0

    print(df['climb'])
    print(df.columns)
    time = (df['timestamp'].diff()/ np.timedelta64(1, 's')).to_numpy()
    time[0]=0
    time=np.cumsum(time)
    climb = df['climb'].to_numpy()
    print(time)

    climb_fft = fft(climb)
    climb_freq = np.fft.fftfreq(len(climb), time[1]-time[0])
    signal2_magnitude = np.abs(climb_fft / len(climb))
    signal2_db = 20 * np.log10(signal2_magnitude)

    plt.plot(climb_freq, np.abs(signal2_db[0:len(climb_fft//2)]))
    plt.xlabel('Frequency (Hz)')
    plt.xlim(left=-0.01,right=0.4)
    plt.ylabel('Amplitude')
    plt.title('FFT of Vertical Position')
    # plt.show()

    position_lat = df['Vposition_lat'].to_numpy()
    position_long = df['Vposition_long'].to_numpy()

    position_lat_fft = fft(position_lat)
    position_long_fft = fft(position_long)
    position_freq = np.fft.fftfreq(len(position_lat), time[1]-time[0])

    signal1_magnitude = np.abs(position_long_fft / len(position_lat))
    position_long_fft_db = 20 * np.log10(signal1_magnitude)
    df['long_fft_db']=position_long_fft_db

    signal1_magnitude = np.abs(position_lat_fft / len(position_lat))
    position_lat_fft_db = 20 * np.log10(signal1_magnitude)
    df['lat_fft_db']=position_lat_fft_db

    plt.scatter(position_freq, np.abs(position_lat_fft_db), c='red')
    plt.scatter(position_freq, np.abs(position_long_fft_db), c='blue')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(left=-0.01)
    plt.ylabel('Amplitude')
    plt.title('FFT of Latitude and Longitude Positions')
    plt.grid()
    ax,ax0=plot_segments_and_trail(df)
    plt.show()

