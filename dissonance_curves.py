# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "sounddevice",
# ]
# ///
"""Real-time Overtone Extraction and Dissonance Curve Visualization.

Captures audio from microphone, extracts overtones/partials using FFT peak detection,
and displays real-time dissonance curves using the Plomp-Levelt model.

References:
- Dissonance model: https://gist.github.com/endolith/3066664
- Frequency estimation: https://gist.github.com/endolith/255291
- Parabolic interpolation: https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html

Usage:
    uv run dissonance_curves.py              # Use default microphone
    uv run dissonance_curves.py -d 1         # Use specific device
    uv run dissonance_curves.py --list-devices
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from scipy.signal.windows import blackmanharris
from scipy.fft import fft
from scipy.signal import find_peaks


# Standard consonant intervals for reference lines
CONSONANT_INTERVALS = [(1, 1), (6, 5), (5, 4), (4, 3), (3, 2), (5, 3), (2, 1)]


# =============================================================================
# FFT and Spectrum
# =============================================================================


def compute_spectrum(audio, window):
    """Apply window and compute FFT magnitude spectrum in dB.

    Args:
        audio: Audio samples (length must match window)
        window: Window function array

    Returns:
        magnitude_db: Magnitude spectrum in dB (positive frequencies only)
    """
    windowed = audio * window
    fft_result = fft(windowed)
    magnitude = np.abs(fft_result)[: len(audio) // 2]
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    return magnitude_db


# =============================================================================
# Noise Reduction
# =============================================================================


def apply_highpass(magnitude_db, frequencies, cutoff_hz=80.0):
    """Zero out frequencies below cutoff to remove low-frequency rumble.

    Args:
        magnitude_db: Magnitude spectrum in dB
        frequencies: Array of frequency bin centers (Hz)
        cutoff_hz: Frequencies below this are set to floor

    Returns:
        Filtered magnitude spectrum in dB
    """
    result = magnitude_db.copy()
    result[frequencies < cutoff_hz] = -100.0
    return result


def compute_noise_threshold(noise_frames, std_multiplier=2.5):
    """Compute noise floor threshold from collected silent frames.

    Args:
        noise_frames: List of magnitude_db arrays captured during silence
        std_multiplier: How many std deviations above mean (higher = more aggressive)

    Returns:
        threshold: Per-bin noise threshold (mean + std_multiplier*std)
    """
    stacked = np.array(noise_frames)
    return np.mean(stacked, axis=0) + std_multiplier * np.std(stacked, axis=0)


def apply_spectral_gate(magnitude_db, threshold, knee_width=6.0):
    """Apply soft-knee spectral gate to reduce noise.

    Args:
        magnitude_db: Input magnitude spectrum in dB
        threshold: Per-bin noise threshold in dB
        knee_width: Width of soft transition in dB

    Returns:
        Gated magnitude spectrum in dB
    """
    diff = magnitude_db - threshold
    mask = np.clip(diff / knee_width, 0, 1)
    floor_db = -100.0
    return magnitude_db * mask + floor_db * (1 - mask)


# =============================================================================
# Peak Detection and Partial Extraction
# =============================================================================


def parabolic_interpolation(magnitude_db, peak_index):
    """Refine peak location using parabolic (quadratic) interpolation.

    Fits a parabola through three points around a peak to find the true
    peak location with sub-bin accuracy.

    Args:
        magnitude_db: Magnitude spectrum in dB
        peak_index: Index of detected peak

    Returns:
        (delta, interpolated_magnitude): Offset from peak_index and refined magnitude
    """
    if peak_index <= 0 or peak_index >= len(magnitude_db) - 1:
        return 0.0, magnitude_db[peak_index]

    alpha = magnitude_db[peak_index - 1]
    beta = magnitude_db[peak_index]
    gamma = magnitude_db[peak_index + 1]

    denominator = alpha - 2 * beta + gamma
    if abs(denominator) < 1e-10:
        return 0.0, beta

    delta = 0.5 * (alpha - gamma) / denominator
    interpolated_mag = beta - 0.25 * (alpha - gamma) * delta
    return delta, interpolated_mag


def extract_partials(
    frequencies, magnitude_db, n_partials=10, min_freq=80.0, max_freq=5000.0, prominence_db=10.0, min_height_db=-60.0
):
    """Extract the N strongest partials from a magnitude spectrum.

    Args:
        frequencies: Array of frequency bin centers (Hz)
        magnitude_db: Magnitude spectrum in dB
        n_partials: Maximum number of partials to extract
        min_freq: Minimum frequency to consider (Hz)
        max_freq: Maximum frequency to consider (Hz)
        prominence_db: Minimum peak prominence in dB
        min_height_db: Minimum peak height in dB

    Returns:
        List of (frequency, linear_amplitude) tuples, sorted by frequency
    """
    bin_width = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    distance_bins = max(1, int(27.0 / bin_width))  # ~27 Hz minimum spacing

    # Restrict to frequency range
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    freq_indices = np.where(freq_mask)[0]
    if len(freq_indices) == 0:
        return []

    # Find peaks
    local_magnitude = magnitude_db[freq_indices]
    peaks, _ = find_peaks(local_magnitude, prominence=prominence_db, distance=distance_bins, height=min_height_db)
    if len(peaks) == 0:
        return []

    # Refine peaks with parabolic interpolation
    global_peaks = freq_indices[peaks]
    partials = []
    for peak_idx in global_peaks:
        delta, interp_mag_db = parabolic_interpolation(magnitude_db, peak_idx)
        interp_freq = frequencies[peak_idx] + delta * bin_width
        linear_amp = 10 ** (interp_mag_db / 20)
        partials.append((interp_freq, linear_amp, interp_mag_db))

    # Take top N by magnitude, then sort by frequency
    partials.sort(key=lambda x: x[2], reverse=True)
    partials = partials[:n_partials]
    partials.sort(key=lambda x: x[0])

    return [(freq, amp) for freq, amp, _ in partials]


# =============================================================================
# Dissonance Calculation (Plomp-Levelt Model)
# =============================================================================


def compute_dissonance(frequencies, amplitudes):
    """Calculate sensory dissonance using the Plomp-Levelt roughness model.

    Args:
        frequencies: Array of partial frequencies (Hz)
        amplitudes: Array of partial amplitudes (linear)

    Returns:
        Total dissonance value (unnormalized)
    """
    if len(frequencies) < 2:
        return 0.0

    # Sort by frequency
    sort_idx = np.argsort(frequencies)
    freqs = np.asarray(frequencies)[sort_idx]
    amps = np.asarray(amplitudes)[sort_idx]

    # Plomp-Levelt parameters
    Dstar, S1, S2 = 0.24, 0.0207, 18.96
    C1, C2 = 5, -5
    A1, A2 = -3.51, -5.75

    # All pairwise combinations
    idx = np.transpose(np.triu_indices(len(freqs), 1))
    if len(idx) == 0:
        return 0.0

    freq_pairs = freqs[idx]
    amp_pairs = amps[idx]

    Fmin = freq_pairs[:, 0]
    Fdif = freq_pairs[:, 1] - freq_pairs[:, 0]
    S = Dstar / (S1 * Fmin + S2)
    SFdif = S * Fdif

    a = np.amin(amp_pairs, axis=1)  # Use minimum amplitude (beat frequency model)
    return np.sum(a * (C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)))


def compute_dissonance_curve(partials, ratios):
    """Compute dissonance curve by sweeping frequency ratios.

    For each ratio, transposes the partials and computes dissonance
    between original and transposed versions.

    Args:
        partials: List of (frequency, amplitude) tuples
        ratios: Array of frequency ratios to evaluate

    Returns:
        Normalized dissonance values for each ratio
    """
    if len(partials) < 2:
        return np.zeros(len(ratios))

    frequencies = np.array([p[0] for p in partials])
    amplitudes = np.array([p[1] for p in partials])

    # Normalize amplitudes
    max_amp = np.max(amplitudes)
    if max_amp > 0:
        amplitudes = amplitudes / max_amp

    # Compute dissonance at each ratio
    dissonance = np.zeros(len(ratios))
    for i, ratio in enumerate(ratios):
        combined_freq = np.concatenate([frequencies, frequencies * ratio])
        combined_amp = np.concatenate([amplitudes, amplitudes])
        dissonance[i] = compute_dissonance(combined_freq, combined_amp)

    # Normalize
    max_diss = np.max(dissonance)
    if max_diss > 0:
        dissonance = dissonance / max_diss

    return dissonance


def run(device=None, sample_rate=44100, fft_size=8192, block_size=2048, noise_frames_needed=16):
    """Run the overtone analyzer with real-time visualization.

    Args:
        device: Audio input device index (None for default)
        sample_rate: Sample rate in Hz
        fft_size: FFT window size (larger = better frequency resolution)
        block_size: Audio block size for callbacks
        noise_frames_needed: Number of frames to collect for noise profiling
    """
    # Precompute constants
    window = blackmanharris(fft_size, sym=False)
    frequencies = np.fft.fftfreq(fft_size, 1 / sample_rate)[: fft_size // 2]
    ratios = np.linspace(1.0, 2.0, 200)

    # Shared state
    state = {
        "buffer": np.zeros(fft_size),
        "noise_frames": [],
        "noise_threshold": None,
        "smoothed_spectrum": None,  # For temporal smoothing
    }

    # Smoothing factor (0 = no smoothing, 1 = infinite smoothing)
    # 0.7 means 70% previous frame + 30% current frame
    smoothing_alpha = 0.5

    # --- Audio callback ---
    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # Simple rolling buffer (same pattern as fft.py)
        state["buffer"] = np.roll(state["buffer"], -len(indata))
        state["buffer"][-len(indata) :] = indata[:, 0]

    # --- Set up figure ---
    fig, (ax_fft, ax_partials, ax_diss) = plt.subplots(
        3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1, 1.5]}
    )
    fig.suptitle("Real-time Overtone Analysis & Dissonance Curve", fontsize=14)

    # Panel 1: FFT Spectrum
    (line_fft,) = ax_fft.plot([], [], "b-", lw=1, alpha=0.8)
    scatter_peaks = ax_fft.scatter([], [], c="red", s=60, zorder=5)
    line_noise = ax_fft.axhline(y=-60, color="gray", linestyle="--", alpha=0.5)
    ax_fft.set_xlim(0, 5000)
    ax_fft.set_ylim(-80, 0)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude (dB)")
    ax_fft.set_title("FFT Spectrum with Detected Peaks")
    ax_fft.grid(True, alpha=0.3)

    # Panel 2: Extracted Partials
    stem_container = [ax_partials.stem([0], [0], linefmt="C0-", markerfmt="C0o", basefmt=" ")]
    ax_partials.set_xlim(0, 5000)
    ax_partials.set_ylim(0, 1.1)
    ax_partials.set_xlabel("Frequency (Hz)")
    ax_partials.set_ylabel("Relative Amplitude")
    ax_partials.set_title("Extracted Partials")
    ax_partials.grid(True, alpha=0.3)

    # Panel 3: Dissonance Curve
    (line_diss,) = ax_diss.plot([], [], "purple", lw=2)
    for n, d in CONSONANT_INTERVALS:
        ax_diss.axvline(n / d, color="silver", linestyle="-", alpha=0.5)
    ax_diss.set_xlim(1.0, 2.0)
    ax_diss.set_ylim(0, 1.1)
    ax_diss.set_xscale("log")
    ax_diss.set_xlabel("Frequency Ratio")
    ax_diss.set_ylabel("Sensory Dissonance")
    ax_diss.set_title("Dissonance Curve (Plomp-Levelt Model)")
    ax_diss.set_xticks([n / d for n, d in CONSONANT_INTERVALS])
    ax_diss.set_xticklabels([f"{n}/{d}" for n, d in CONSONANT_INTERVALS])
    ax_diss.minorticks_off()

    # Status text
    status_text = fig.text(0.5, 0.01, "Initializing...", ha="center", fontsize=11, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # --- Animation update ---
    def update_plot(_):
        # Compute spectrum
        magnitude_db = compute_spectrum(state["buffer"], window)

        # High-pass filter to remove low-frequency rumble
        magnitude_db = apply_highpass(magnitude_db, frequencies, cutoff_hz=80.0)

        # Noise profiling (collect frames, then compute threshold)
        if len(state["noise_frames"]) < noise_frames_needed:
            state["noise_frames"].append(magnitude_db.copy())
            progress = len(state["noise_frames"]) / noise_frames_needed
            print(f"\rProfiling noise... {progress:.0%}", end="", flush=True)
            status_text.set_text(f"Profiling noise... {progress:.0%}")

            if len(state["noise_frames"]) == noise_frames_needed:
                state["noise_threshold"] = compute_noise_threshold(state["noise_frames"])
                print("\nNoise profiling complete!")

        # Apply spectral gate if threshold is ready
        if state["noise_threshold"] is not None:
            magnitude_db = apply_spectral_gate(magnitude_db, state["noise_threshold"])
            status_text.set_text("Ready - Listening for audio input")

        # Temporal smoothing (exponential moving average)
        if state["smoothed_spectrum"] is None:
            state["smoothed_spectrum"] = magnitude_db.copy()
        else:
            state["smoothed_spectrum"] = (
                smoothing_alpha * state["smoothed_spectrum"] + (1 - smoothing_alpha) * magnitude_db
            )
        magnitude_db = state["smoothed_spectrum"]

        # Extract partials
        partials = extract_partials(frequencies, magnitude_db, n_partials=15)

        # Compute dissonance curve
        dissonance = compute_dissonance_curve(partials, ratios)

        # Update FFT panel
        line_fft.set_data(frequencies, magnitude_db)

        if partials:
            peak_freqs = [p[0] for p in partials]
            peak_mags = [20 * np.log10(p[1] + 1e-10) for p in partials]
            scatter_peaks.set_offsets(np.column_stack([peak_freqs, peak_mags]))
        else:
            scatter_peaks.set_offsets(np.empty((0, 2)))

        if state["noise_threshold"] is not None:
            mean_threshold = np.mean(state["noise_threshold"])
            line_noise.set_ydata([mean_threshold, mean_threshold])

        # Update partials panel (stem plot)
        stem_container[0].remove()
        if partials:
            partial_freqs = [p[0] for p in partials]
            partial_amps = [p[1] for p in partials]
            max_amp = max(partial_amps)
            partial_amps_norm = [a / max_amp for a in partial_amps]
            stem_container[0] = ax_partials.stem(
                partial_freqs, partial_amps_norm, linefmt="C0-", markerfmt="C0o", basefmt=" "
            )
        else:
            stem_container[0] = ax_partials.stem([0], [0], linefmt="C0-", markerfmt="C0o", basefmt=" ")

        # Update dissonance curve
        line_diss.set_data(ratios, dissonance)

        return (line_fft, scatter_peaks, line_diss, status_text)

    def init_plot():
        line_fft.set_data([], [])
        scatter_peaks.set_offsets(np.empty((0, 2)))
        line_diss.set_data([], [])
        return (line_fft, scatter_peaks, line_diss, status_text)

    # --- Start audio and animation ---
    print("Starting Overtone Analyzer...")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  FFT size: {fft_size} ({sample_rate / fft_size:.1f} Hz resolution)")
    print(f"  Update rate: ~{1000 * block_size / sample_rate:.0f} ms")
    print()
    print("Please remain quiet for the first second while noise floor is measured...")
    print()

    stream = sd.InputStream(
        device=device,
        channels=1,
        samplerate=sample_rate,
        callback=audio_callback,
        blocksize=block_size,
    )

    frame_interval = int(1000 * block_size / sample_rate)
    anim = FuncAnimation(
        fig, update_plot, init_func=init_plot, interval=frame_interval, blit=False, cache_frame_data=False
    )

    with stream:
        plt.show()


def test_spectrum_and_partials():
    """Test FFT spectrum computation and partial extraction."""
    print("=" * 60)
    print("Spectrum & Partial Extraction Test")
    print("=" * 60)

    # Setup
    fft_size = 8192
    sample_rate = 44100
    window = blackmanharris(fft_size, sym=False)
    frequencies = np.fft.fftfreq(fft_size, 1 / sample_rate)[: fft_size // 2]
    bin_width = sample_rate / fft_size

    print(f"Frequency resolution: {bin_width:.2f} Hz")
    print(f"Frequency bins: {len(frequencies)}")

    # Generate test signal: 440 Hz fundamental + harmonics
    t = np.linspace(0, fft_size / sample_rate, fft_size)
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t) + 0.25 * np.sin(2 * np.pi * 1320 * t)

    # Compute spectrum and extract partials
    magnitude_db = compute_spectrum(signal, window)
    partials = extract_partials(frequencies, magnitude_db, n_partials=5)

    print("\nTest signal: 440 Hz + 880 Hz (0.5x) + 1320 Hz (0.25x)")
    print("\nDetected partials:")
    for f, a in partials:
        print(f"  {f:.1f} Hz (amplitude: {a:.4f})")

    # Verify expected frequencies are detected
    expected = [440, 880, 1320]
    detected_freqs = [p[0] for p in partials]

    print("\nValidation:")
    for exp in expected:
        closest = min(detected_freqs, key=lambda x: abs(x - exp))
        error = abs(closest - exp)
        status = "PASS" if error < 10 else "FAIL"
        print(f"  Expected {exp} Hz, found {closest:.1f} Hz (error: {error:.1f} Hz) [{status}]")

    return partials


def test_dissonance_curve(partials):
    """Test dissonance curve generation."""
    print("\n" + "=" * 60)
    print("Dissonance Curve Test")
    print("=" * 60)

    ratios = np.linspace(1.0, 2.0, 200)
    dissonance = compute_dissonance_curve(partials, ratios)

    print(f"Curve points: {len(ratios)}")
    print(f"Ratio range: {ratios[0]:.2f} to {ratios[-1]:.2f}")

    # Find global minimum
    min_idx = np.argmin(dissonance)
    print(f"\nGlobal minimum dissonance at ratio: {ratios[min_idx]:.3f}")

    # Check dissonance at consonant intervals
    consonant_intervals = [(1, 1), (6, 5), (5, 4), (4, 3), (3, 2), (5, 3), (2, 1)]

    print("\nDissonance at consonant intervals:")
    for n, d in consonant_intervals:
        ratio = n / d
        idx = np.argmin(np.abs(ratios - ratio))
        print(f"  {n}/{d} = {ratio:.3f}: dissonance = {dissonance[idx]:.3f}")


def test_parabolic_interpolation():
    """Test parabolic peak interpolation."""
    print("=" * 60)
    print("Parabolic Interpolation Test")
    print("=" * 60)

    # Create a simple peak at index 5, slightly off-center
    mag_db = np.array([-20, -18, -15, -10, -5, -2, -3, -8, -15, -20], dtype=np.float64)
    peak_idx = 5

    delta, interp_mag = parabolic_interpolation(mag_db, peak_idx)

    print(f"Peak at index {peak_idx}: {mag_db[peak_idx]:.1f} dB")
    print(f"Interpolated offset: {delta:.3f} bins")
    print(f"Interpolated magnitude: {interp_mag:.3f} dB")
    print(f"True peak location: {peak_idx + delta:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time overtone extraction and dissonance curve visualization")
    parser.add_argument("-d", "--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate in Hz (default: 44100)")
    parser.add_argument("--fft-size", type=int, default=8192, help="FFT size (default: 8192)")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--test", action="store_true", help="Run test routines and exit")
    args = parser.parse_args()

    if args.test:
        test_parabolic_interpolation()
        partials = test_spectrum_and_partials()
        test_dissonance_curve(partials)
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        sys.exit(0)

    if args.list_devices:
        print(sd.query_devices())
    else:
        run(device=args.device, sample_rate=args.sample_rate, fft_size=args.fft_size)
