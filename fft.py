# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "sounddevice",
# ]
# ///
"""Real-time FFT Visualization of Audio

Demo script showing how to work with sounddevice and matplotlib.

Usage:
    uv run fft.py                     # Use microphone input
    uv run fft.py song.wav            # Play and visualize WAV file
    uv run fft.py --list-devices      # List audio devices
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.fftpack import fft


def load_audio(wav_file):
    """Load and normalize WAV file to mono."""
    sample_rate, audio_data = wav.read(wav_file)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(float) / np.iinfo(np.int16).max
    return sample_rate, audio_data


def run(wav_file=None, device=None, sample_rate=44100, buffer_size=2048, block_size=1024, max_freq=None):
    # Load WAV or use mic settings
    if wav_file:
        sample_rate, audio_data = load_audio(wav_file)
        title = f"FFT Visualization: {wav_file}"
    else:
        audio_data = None
        title = "FFT Visualization (Microphone)"

    # FFT setup
    window = np.hanning(buffer_size)
    freqs = np.fft.fftfreq(buffer_size, 1 / sample_rate)
    freq_range = freqs[: buffer_size // 2]

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 6))
    (line,) = ax.plot([], [], "b-", lw=2)
    ax.set_xlim(0, max_freq or freq_range[-1] / 2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)

    # Shared state
    state = {"frame": 0, "buffer": np.zeros(buffer_size)}

    def mic_callback(indata, frames, time, status):
        if status:
            print(status)
        state["buffer"] = np.roll(state["buffer"], -len(indata))
        state["buffer"][-len(indata) :] = indata[:, 0]

    def wav_callback(outdata, frames, time, status):
        if status:
            print(status)
        if state["frame"] + frames > len(audio_data):
            raise sd.CallbackStop()
        chunk = audio_data[state["frame"] : state["frame"] + frames]
        state["buffer"] = np.roll(state["buffer"], -len(chunk))
        state["buffer"][-len(chunk) :] = chunk
        outdata[:] = chunk.reshape(-1, 1)
        state["frame"] += frames

    def update_plot(_):
        windowed = state["buffer"] * window
        magnitude = np.abs(fft(windowed))[: buffer_size // 2]
        if (max_mag := np.max(magnitude)) > 0:
            magnitude /= max_mag
        line.set_data(freq_range, magnitude)
        return (line,)

    def init_plot():
        line.set_data(freq_range, np.zeros_like(freq_range))
        return (line,)

    # Create appropriate stream
    if wav_file:
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            callback=wav_callback,
            blocksize=block_size,
        )
    else:
        stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=sample_rate,
            callback=mic_callback,
            blocksize=block_size,
        )

    frame_interval = int(1000 * block_size / sample_rate)
    anim = FuncAnimation(fig, update_plot, init_func=init_plot, interval=frame_interval, blit=True)

    with stream:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time FFT visualization of audio")
    parser.add_argument("wav_file", nargs="?", default=None, help="Path to WAV file (uses microphone if omitted)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (mic mode only)")
    parser.add_argument(
        "--sample-rate", type=int, default=44100, help="Sample rate in Hz (mic mode only, default: 44100)"
    )
    parser.add_argument("--buffer-size", type=int, default=2048, help="FFT buffer size (default: 2048)")
    parser.add_argument("--block-size", type=int, default=1024, help="Audio block size (default: 1024)")
    parser.add_argument("--max-freq", type=float, default=None, help="Max frequency to display (Hz)")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
    else:
        run(args.wav_file, args.device, args.sample_rate, args.buffer_size, args.block_size, args.max_freq)
