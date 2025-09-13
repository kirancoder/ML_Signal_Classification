# ML_Signal_Classification
Signal feature extraction and classification using ML classifier
 
## 📘 Feature Extraction

This notebook extracts a comprehensive set of features from time-series signals using:

- **Time-domain analysis**
- **Frequency-domain analysis**
- **Autocorrelation**
- **Empirical Mode Decomposition (EMD)**
- **Hilbert-Huang Transform (HHT)**

These features are useful for classification, anomaly detection, and understanding signal behavior in mechanical, biomedical, and other physical systems.

### 🕒 Time-Domain Features

| Feature | Description |
|--------|-------------|
| `mean` | Average value of the signal. Indicates baseline level. |
| `std` | Standard deviation. Measures variability or spread. |
| `min`, `max` | Minimum and maximum values. Show signal range. |
| `skew` | Asymmetry of the signal distribution. Positive skew = longer tail on right. |
| `kurtosis` | "Tailedness" of the distribution. High kurtosis = more outliers. |
| `energy` | Sum of squared values. Represents signal power. |
| `peak_to_peak` | Difference between max and min. Measures amplitude range. |
| `zero_crossings` | Number of times the signal crosses zero. Indicates frequency content or oscillatory nature. |

### 📊 Frequency-Domain Features (via Welch PSD)

| Feature | Description |
|--------|-------------|
| `dominant_freq` | Frequency with highest power. Often the main oscillation. |
| `spectral_entropy` | Complexity of the spectrum. Higher = more uniform distribution (noisy). |
| `spectral_centroid` | "Center of mass" of the spectrum. Indicates where most energy is concentrated. |
| `spectral_bandwidth` | Spread of the spectrum around the centroid. |
| `spectral_flatness` | Ratio of geometric mean to arithmetic mean of PSD. High = noise-like, low = tone-like. |

### 🔁 Autocorrelation Feature

| Feature | Description |
|--------|-------------|
| `autocorr_peak` | Peak of autocorrelation (excluding lag 0), normalized. Measures periodicity or self-similarity. |

### ⚡ FFT Features (Fast Fourier Transform)

| Feature       | Description |
|---------------|-------------|
| `fft_max`     | Maximum amplitude in the FFT spectrum. Indicates the strongest frequency component. |
| `fft_mean`    | Mean amplitude of the FFT spectrum. Reflects average spectral energy. |
| `fft_std`     | Standard deviation of FFT amplitudes. Measures spread or variability in frequency content. |
| `fft_entropy` | Entropy of the normalized FFT spectrum. Higher values indicate more complex or noisy signals. |


### 🌊 EMD-HHT Features

These are derived from **Empirical Mode Decomposition (EMD)** and **Hilbert Transform**:

| Feature | Description |
|--------|-------------|
| `num_imfs` | Number of Intrinsic Mode Functions (IMFs) extracted. More IMFs = more complexity. |
| `total_imf_energy` | Sum of energy across all IMFs. Indicates how much signal is captured by decomposition. |
| `mean_inst_freq` | Average instantaneous frequency across IMFs. Captures dynamic frequency behavior. |
| `mean_inst_amp` | Average amplitude envelope across IMFs. Reflects signal strength over time. |


