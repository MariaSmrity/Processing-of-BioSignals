import numpy as np
import scipy.io
import matplotlib.pyplot as plt

num_iterations = 50

def gradient_lms(S_chest, S_abdomen, delta, N):
    """
    Gradient LMS adaptive filtering.
    
    Parameters:
    - S_chest: Input signal from chest.
    - S_abdomen: Input signal from abdomen.
    - delta: Step size for coefficient update.
    - N: Filter order.
    
    Returns:
    - e: Filtered output signal.
    - H: Filter coefficients.
    """
    # Initialize filter coefficients
    H = np.zeros(N)
    
    # Initialize filtered output signal
    e = np.zeros_like(S_chest)
    
    # Iterate through each sample
    for i in range(N, len(S_chest)):
        # Extract input vector X(n+1)
        X = S_abdomen[i-N:i]
        
        # Compute filtered output y(n+1)
        y = np.dot(H, X)
        
        # Compute error e(n+1)
        e[i] = S_chest[i] - y
        
        # Update filter coefficients
        H = H + delta * X * e[i]
    
    return e, H

# Load data
data = scipy.io.loadmat('Parent_Fetus_ECGs.mat')
S_chest = data['S_chest'].flatten()
S_abdomen = data['S_abdomen'].flatten()

# Plot raw signals
plt.figure()
plt.plot(S_chest[2000:2500], label='Chest Signal')
plt.plot(S_abdomen[2000:2500], label='Abdomen Signal')
plt.title('Raw Signals')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Parameters
N = 5 # Filter order (odd number)
delta = 1E-6  # Step size for coefficient update


# Iterate for multiple iterations
for iter in range(num_iterations):
    # Apply gradient LMS adaptive filtering
    e, H = gradient_lms(S_chest, S_abdomen, delta, N)

    # Plot filtered output signal
    plt.figure()
    plt.plot(e[2000:2500])
    plt.title('Filtered Output Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Plot filter's frequency response
    plt.figure()
    plt.plot(np.abs(np.fft.fft(H, 1024)))
    plt.title('Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()