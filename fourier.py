import numpy as np
import matplotlib.pyplot as plt


concentrations = [0, 100, 1000]
heart_rates = [[1.73, 1.13, 1.26], [1.53, 1.33, 1.33], [0.8, 0.73, 0.53]]

# The number of times heartbeats in 60 seconds = 72 times
# 72/60 = 1.25 times heartbeats per second
relative_heart_rates = [[(h / 1.25) * 100 for h in r] for r in heart_rates]
print(relative_heart_rates)

for i, rates in enumerate(relative_heart_rates):
    plt.scatter(concentrations, rates, marker='o', label=f'Organoid {i+1}')

plt.xlabel('Concentration (nM)')
plt.ylabel('Relative Heart Rate (%)')
plt.title('Nifedifine')
plt.legend()
plt.xticks(concentrations)
plt.savefig('Nifedifine relative sample.png')
plt.show()

'''
# Generate a sample signal
time = np.linspace(0, 15, 450)  # Create a time array from 0 to 15 with 450 points
frequency = 2  # Frequency of the signal in Hz
# sin(2πft) fundamental formula in signal processing and represents a simple harmonic oscillation.
signal = np.sin(2 * np.pi * frequency * time)

# Perform Fourier Transform
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(time), time[1] - time[0])
print(time[1] - time[0])
print(len(time))
# print(frequencies)

# Plot the original signal and its Fourier Transform
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(time, signal)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(122)
plt.plot(frequencies, np.abs(fft_result))
plt.title('Fourier Transform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 100)  # Only show frequencies up to 100 Hz

plt.tight_layout()
plt.show()


# Find the index of the maximum amplitude in the FFT result
dominant_frequency_index = np.argmax(np.abs(fft_result))
dominant_frequency = frequencies[dominant_frequency_index]

print(dominant_frequency)
'''

