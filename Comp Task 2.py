import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin, freqz

# Load EEG data from the .mat file
eeg_data = loadmat('case_eeg.mat')


# Extract the EEG signal and annotation signals from the loaded data
eeg_signal = eeg_data['eeg_ch']  # Assuming 'EEG' is the variable name containing the EEG data
before_annotations = eeg_data['before']  # Assuming 'before' is the variable name for before annotations
after_annotations = eeg_data['after']    # Assuming 'after' is the variable name for after annotations
eeg_time = eeg_data['eeg_time']
# Multiply the before and after annotation signals by a factor to make them stand out
before_annotations *= 100
after_annotations *= 100

# Sampling frequency of the EEG signal (Hz)
fs_eeg = 1 / (eeg_time[1] - eeg_time[0])
fs = fs_eeg  # Assuming sampling frequency is 1000 Hz


# Calculate the time vector
t = np.arange(0, len(eeg_signal) / fs, 1/fs)

# Reshape annotation signals to match the length of the time vector
before_annotations_resized = np.repeat(before_annotations, len(t) // len(before_annotations), axis=0)
after_annotations_resized = np.repeat(after_annotations, len(t) // len(after_annotations), axis=0)

# Plot the EEG signal
plt.figure(figsize=(10, 6))
plt.plot(t, eeg_signal, label='EEG Signal')

# Plot the before and after annotation signals
plt.plot(t, before_annotations_resized, label='Before Annotations', alpha=0.5)
plt.plot(t, after_annotations_resized, label='After Annotations', alpha=0.5)

# Set labels and title
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Signal with Before and After Annotations')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

#========== Part 2 ===========

# Design FIR highpass filter
cutoff_freq = 0.5  # Cutoff frequency of the highpass filter in Hz
num_taps = 101  # Number of filter taps
b = firwin(num_taps, cutoff_freq / (fs / 2), pass_zero=False)

# Plot frequency response of the filter
w, h = freqz(b, worN=8000)
plt.figure()
plt.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)))
plt.axvline(cutoff_freq, color='r')  # Cutoff frequency
plt.title("FIR Highpass Filter Frequency Response")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid()
plt.show()

# Apply the filter to the EEG signal
eeg_filtered = filtfilt(b, [1.0], eeg_signal, method="pad", axis = 0)

# Plot the original and filtered EEG signals
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(eeg_signal)) / fs, eeg_signal, label='Original EEG Signal')
plt.plot(np.arange(len(eeg_filtered)) / fs, eeg_filtered, label='Filtered EEG Signal')
plt.title('Original and Filtered EEG Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
#===================== Part 3

def spectral_entropy(P, nfactor):
    C_n = 1 / np.sum(P)
    P_n = C_n * P
    log_P_n = -np.log(P_n)
    H = np.transpose(P_n) @ log_P_n / nfactor
    return H

def entropy_eeg(eeg, fs):
    eeg_length = len(eeg)
    wlen = 5  # window length to calculate PSD over (seconds)
    overlap = 50  # percentage overlap for windows
    H_length = int(np.round(1 + eeg_length / (wlen * fs * ((100 - overlap) / 100))))
    H = np.zeros(H_length) * np.nan  # initialization array containing spectral entropy values
    t = np.zeros(H_length) * np.nan  # initialization array containing time instances at which spectral entropy values are calculated
    
    # spectrum calculation over first window
    t0 = 0  # first sample of the window
    t1 = int(wlen * fs)  # last sample
    step = int(np.round(fs * wlen * (100 - overlap) / 100))  # number of samples to shift the window each step
    cnt = 0  # step counter
    
    while t1 < eeg_length:  # as long as the last sample of the window is not beyond the EEG signal end
        f, PSD = welch(eeg[t0:t1], fs=fs, window='hann', nperseg=t1 - t0 + 1, noverlap=0)  # calculate the PSD over a window
        f_low = 0.8  # lower limit of frequency range of interest
        f_high = 32.0  # upper limit of frequency range of interest
        ind_f = np.where((f >= f_low) & (f <= f_high))[0]  # find indexes of relevant frequencies in f array
        sf = np.log(len(ind_f))  # scaling factor, needed for normalization of spectral entropy
        
        t[cnt] = t1 / fs  # time instance of PSD (and entropy calculation), defined as end of window
        H[cnt] = spectral_entropy(PSD[ind_f], sf)  # calculate normalized spectral entropy over ind_f freq band
        t0 += step  # update start position of window
        t1 += step  # update end position of window
        cnt += 1  # update step counter
        
    return H, t

#-------------- Part 4
H, H_t = entropy_eeg(eeg_filtered, fs_eeg)

# Extract the required signals from the loaded data
bis_ch = eeg_data['bis_ch']  # 'bis_ch' is the variable name for bispectral entropy
bis_time = eeg_data['bis_time']
annot_time = eeg_data['annot_time']
H_scaled = 100 * H

# Plot all signals together
plt.figure(figsize=(12, 8))

# Plot bis_ch
plt.plot(bis_time, bis_ch, label='Bispectral Entropy (bis_ch)', color='blue')

# Plot scaled spectral entropy (H_scaled)
plt.plot(H_t, H_scaled, label='Scaled Spectral Entropy (100*H)', color='green')

# Plot before and after annotations
plt.plot(annot_time, before_annotations, label='Before Annotations', color='red', alpha=0.5)
plt.plot(annot_time, after_annotations, label='After Annotations', color='orange', alpha=0.5)

# Set plot title and labels
plt.title('Comparison of Spectral Entropy, Bispectral Entropy, Before and After Annotations')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()