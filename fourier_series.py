import numpy as np
from code_functions import DF_to_segmented_DF,plot_segments_and_trail,print_df
import pandas as pd
import matplotlib.pyplot as plt

def plot_segments(wdf, coeffs):
    t = np.linspace(0, 1, coeffs.shape[0])
    fig = plt.figure()
    ax0=fig.add_subplot(2,1,1)
    ax1=fig.add_subplot(2,1,2)
    ax0.plot(wdf['lat'],wdf['long'], label='Original')
    ax1.plot(t, coeffs, label='Reconstructed')
    plt.legend()
    plt.show()
def sample_curve(curve, num_samples):
    """
    Sample a curve into a set of points.
    """
    t = np.linspace(0, 1, num_samples)
    samples = np.array([curve(ti) for ti in t])
    return samples

def approximate_curve(x,y, n_coeffs):
    """
    Approximate a curve using Fourier series.
    """
    samples = np.column_stack([x, y])
    n = samples.shape[0]
    t = np.linspace(0, 1, n)
    
    def f(x):
        """
        The curve to be approximated.
        """
        return np.interp(x, t, samples[:, 0])
    
    def a_k(k):
        """
        The Fourier coefficient for the cosine term.
        """
        return 2 / n * np.sum(np.cos(2 * np.pi * k * t) * f(t))
    
    def b_k(k):
        """
        The Fourier coefficient for the sine term.
        """
        return 2 / n * np.sum(np.sin(2 * np.pi * k * t) * f(t))
    
    coeffs = []
    for k in range(n_coeffs):
        coeffs.append((a_k(k), b_k(k)))
    
    def approx(x):
        """
        The Fourier series approximation.
        """
        result = np.zeros_like(x)
        for k, (ak, bk) in enumerate(coeffs):
            result += ak * np.cos(2 * np.pi * k * x) + bk * np.sin(2 * np.pi * k * x)
        return result
    
    return approx(x)
def dominant_frequencies(x,y, num_frequencies=3, sampling_rate=1):
    """
    Find the num_frequencies dominant frequencies in the input signal.
    """
    samples = np.column_stack([x,y])
    n = len(samples)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    fft = np.fft.fft(samples)
    fft_abs = np.abs(fft)

    sorted_indices = np.argsort(fft_abs)[::-1][:num_frequencies]
    dominant_frequencies = freq[sorted_indices]
    
    return dominant_frequencies
filename = 'data/221005_eksempelsegment001.xlsx'
# Read data
TDF0 = DF_to_segmented_DF(pd.read_excel(filename))
TDF1 = DF_to_segmented_DF(pd.read_parquet('data/2022-06-05-12-12-09 (1).parquet')).iloc[0:1000]
# Look at data
# plot_segments_and_trail(TDF0,Show_Plot=True)
# We choose to look at segments 1 and 2
# plot_segments_and_trail(TDF1,Show_Plot=True)
chosen_segments = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
fourier=[]
for i in chosen_segments:
    wdf = TDF1.loc[TDF1['segments']==i].copy(deep = True)
    wdf['lat'] = wdf['Vposition_lat'].cumsum()
    wdf['long'] = wdf['Vposition_long'].cumsum()
    wdf['newaltitude'] = wdf['Valtitude'].cumsum()
    coeffs = approximate_curve(wdf['long'], wdf['lat'], n_coeffs=3)
    fourier.append(coeffs)
    # print(coeffs.shape)
    freqs = dominant_frequencies(wdf['long'], wdf['lat'])*100
    # print(i,coeffs)
    division = abs(wdf['lat']/wdf['long'])
    division[0]=0 
    print(division)
    print(sum(division),np.mean(division))
    plot_segments(wdf,coeffs)
    # reconstructed_segment = approximate_curve(wdf['long'],wdf['lat'], num_coeffs=3)
    # plot_segments(np.column_stack(wdf['long'],wdf['lat']), reconstructed_segment)
# print(fourier)