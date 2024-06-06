from rtlsdr import RtlSdr
from scipy.fft import fft, fftshift
import numpy as np
import pylab
from pylab import *
import matplotlib.pyplot as plt

plt.ion()

sdr = RtlSdr()
num_samples = 1024 ** 2

def next_power_2(num):
    current_result = 0
    iterations = 0
    
    while(current_result < num):
        iterations += 1
        current_result = 1 << iterations
        
    return iterations

def fix_iq_imbalance(x):
    # stolen from reddit
    # remove DC and save input power
    z = x - mean(x)
    p_in = var(z)

    # scale Q to have unit amplitude (remember we're assuming a single input tone)
    Q_amp = sqrt(2*mean(x.imag**2))
    z /= Q_amp

    I, Q = z.real, z.imag

    alpha_est = sqrt(2*mean(I**2))
    sin_phi_est = (2/alpha_est)*mean(I*Q)
    cos_phi_est = sqrt(1 - sin_phi_est**2)

    I_new_p = (1/alpha_est)*I
    Q_new_p = (-sin_phi_est/alpha_est)*I + Q

    y = (I_new_p + 1j*Q_new_p)/cos_phi_est

    print('phase error:', arccos(cos_phi_est)*360/2/pi, 'degrees')
    print('amplitude error:', 20*log10(alpha_est), 'dB')

    return y*sqrt(p_in/var(y))

def plot_fft(time_domain_data):
    samples = fix_iq_imbalance(time_domain_data)

    fft_data = fftshift(fft(samples))
    plt.clf()

    plt.plot(fft_data)
    plt.draw()
    #pylab.psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
    plt.pause(0.0001)
    

def fm_demod(x, df=1.0, fc=0.0):
    # stolen from reddit
    ''' Perform FM demodulation of complex carrier.

    Args:
        x (array):  FM modulated complex carrier.
        df (float): Normalized frequency deviation [Hz/V].
        fc (float): Normalized carrier frequency.

    Returns:
        Array of real modulating signal.
    '''

    # Remove carrier.
    n = np.arange(len(x))
    rx = x*np.exp(-1j*2*np.pi*fc*n)

    # Extract phase of carrier.
    phi = np.arctan2(np.imag(rx), np.real(rx))

    # Calculate frequency from phase.
    y = np.diff(np.unwrap(phi)/(2*np.pi*df))

    return y


sdr.center_freq = 107.9e6
sdr.sample_rate = 2.4e6
sdr.freq_correction = 60
sdr.set_bandwidth(2000)
sdr.gain = 30


# plt.xlabel("Frequency (mHz)")
# plt.ylabel("Relative Power (dB)")

for x in range(10):
    plot_fft(sdr.read_samples(num_samples))
    #print(fm_demod())

sdr.close()