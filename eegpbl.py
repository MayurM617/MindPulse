import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics.pairwise import euclidean_distances

# Load EEG dataset
file_path = "C:/Users/Hp/OneDrive/Desktop/eegdataset/emotions.csv"
data = pd.read_csv(file_path)

# Rename labels
data['label'] = data['label'].replace({
    'neutral': 'normal',
    'positive': 'happy',
    'negative': 'sad'
})

# Extract FFT feature columns
fft_cols = [col for col in data.columns if 'fft' in col]

# Define EEG frequency bands
eeg_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 100)
}

# Sampling frequency
fs = 256

# ---------------- USER INPUT ----------------
user_input = input("\nEnter your EEG signal values (comma or space separated):\n")

if ',' in user_input:
    eeg_signal = np.array([float(val) for val in user_input.split(',')])
else:
    eeg_signal = np.array([float(val) for val in user_input.split()])

# Remove DC offset
eeg_signal -= np.mean(eeg_signal)

# Get frequencies and power of user input
frequencies, power = welch(eeg_signal, fs, nperseg=min(1024, len(eeg_signal)))

# Calculate band powers for input
user_band_power = []
for band, (low, high) in eeg_bands.items():
    idx = np.where((frequencies >= low) & (frequencies <= high))
    band_power = np.trapz(power[idx], frequencies[idx])
    user_band_power.append(band_power)

# ---------------- DATASET BAND POWERS ----------------
samples_band_power = []
labels = []

for _, row in data.iterrows():
    signal = row[fft_cols].values.astype(float)
    signal -= np.mean(signal)
    freqs, pxx = welch(signal, fs, nperseg=1024)

    powers = []
    for band, (low, high) in eeg_bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))
        band_power = np.trapz(pxx[idx], freqs[idx])
        powers.append(band_power)

    samples_band_power.append(powers)
    labels.append(row['label'])

samples_band_power = np.array(samples_band_power)

# ---------------- PREDICTION ----------------
user_vector = np.array(user_band_power).reshape(1, -1)
distances = euclidean_distances(user_vector, samples_band_power)
closest_idx = np.argmin(distances)
predicted_emotion = labels[closest_idx]

print(f"\n🧠 Predicted Emotion: **{predicted_emotion.upper()}**")

# ---------------- PLOTTING ----------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Power Spectrum
axs[0].semilogy(frequencies, power)
axs[0].set_title("Power Spectrum (User Input)")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel("Power (µV²/Hz)")
axs[0].grid(True)

# Plot 2: Band Powers Across Emotions (averaged)
grouped = {}
for i, label in enumerate(labels):
    if label not in grouped:
        grouped[label] = []
    grouped[label].append(samples_band_power[i])

avg_by_emotion = {label: np.mean(grouped[label], axis=0) for label in grouped}

bands = list(eeg_bands.keys())
x = np.arange(len(bands))
width = 0.25

for i, (label, power_vals) in enumerate(avg_by_emotion.items()):
    axs[1].bar(x + i * width, power_vals, width, label=label)

axs[1].set_xticks(x + width)
axs[1].set_xticklabels(bands)
axs[1].set_xlabel("EEG Bands")
axs[1].set_ylabel("Avg Band Power (µV²)")
axs[1].set_title("EEG Band Power by Emotion (Dataset)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
