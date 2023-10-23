import numpy as np
import matplotlib.pyplot as plt

# Generate a sample signal
time = np.linspace(0, 1, 1000)  # Create a time array from 0 to 1 with 1000 points
frequency = 50  # Frequency of the signal in Hz
signal = np.sin(2 * np.pi * frequency * time)

# Perform Fourier Transform
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(time), time[1] - time[0])
print(frequencies)

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
