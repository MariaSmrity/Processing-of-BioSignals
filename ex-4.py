import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = scipy.io.loadmat('data_ca4.mat')
heart_beats = data['heart_beats']

# Extract necessary variables
starting_beat_annotation = heart_beats[0]
ending_beat_annotation = heart_beats[-1]
total_ECG_sample_points = ending_beat_annotation - starting_beat_annotation
Fs = 128

# Calculate RR intervals and time points
RR_Intervals = np.zeros(len(heart_beats)-1)
x = np.zeros(len(heart_beats)-1)
for i in range(1, len(heart_beats)):
    RR_Intervals[i-1] = (heart_beats[i]-heart_beats[i-1])/Fs
    x[i-1] = heart_beats[i]/Fs

# Plotting data
idx1 = np.argmin(np.abs(x - 3600*6))
idx2 = np.argmin(np.abs(x - 3600*12))
idx3 = np.argmin(np.abs(x - 3600*18))

plt.figure(figsize=(10, 8))
plt.subplot(4,1,1)
plt.plot(x[0:idx1]/3600, RR_Intervals[0:idx1])
plt.xlim([x[1]/3600, x[idx1]/3600])
plt.ylim([0.25, 2])
plt.ylabel('RR Interval (s)')

plt.subplot(4,1,2)
plt.plot(x[idx1+1:idx2]/3600, RR_Intervals[idx1+1:idx2])
plt.xlim([x[idx1+1]/3600, x[idx2]/3600])
plt.ylim([0.25, 8])
plt.ylabel('RR Interval (s)')

plt.subplot(4,1,3)
plt.plot(x[idx2+1:idx3]/3600, RR_Intervals[idx2+1:idx3])
plt.xlim([x[idx2+1]/3600, x[idx3]/3600])
plt.ylim([0.25, 2])
plt.ylabel('RR Interval (s)')

plt.subplot(4,1,4)
plt.plot(x[idx3+1:]/3600, RR_Intervals[idx3+1:])
plt.xlim([x[idx3+1]/3600, 24])
plt.ylim([0.25, 6])
plt.ylabel('RR Interval (s)')
plt.xlabel('Time (hours)')
plt.show()

# Calculate SDNN
SDNN = np.std(RR_Intervals)
print("SDNN:", SDNN)

# Calculate rMSSD
SD = np.diff(RR_Intervals)
rMSSD = np.sqrt(np.mean(SD ** 2))
print("rMSSD:", rMSSD)

# Calculate pNN50
NN50 = np.sum(np.abs(SD) > 0.050)
pNN50 = NN50/len(SD)*100
print("pNN50:", pNN50)

# RR Interval Histogram
plt.figure()
plt.hist(RR_Intervals*10000, bins=30)
plt.xlabel('RR Interval (ms)')
plt.ylabel('Frequency')
plt.title('RR Interval Histogram')
plt.show()

# Poincaré plot
plt.figure()
plt.scatter(RR_Intervals[:-1], RR_Intervals[1:])
plt.xlabel('RR Interval (s) - Current Beat')
plt.ylabel('RR Interval (s) - Next Beat')
plt.title('Poincaré Plot of RR Intervals')
plt.show()
# Calculate SDANN
seg_start_idx = 0
Average_5_min_signal = []
for i in range(1, int(np.ceil(total_ECG_sample_points/(Fs*60*5)))+1):
    if i == int(np.ceil(total_ECG_sample_points/(Fs*60*5))):
        Average_5_min_signal.append(np.mean(RR_Intervals[seg_start_idx+1:]))
    else:
        seg_end_idx = np.argmin(np.abs(x - 300*i))
        Average_5_min_signal.append(np.mean(RR_Intervals[seg_start_idx+1:seg_end_idx]))
        seg_start_idx = seg_end_idx

SDANN = np.std(Average_5_min_signal)
print("SDANN:", SDANN)

#plt.show()
