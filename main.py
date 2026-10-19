import numpy as np
from dataclasses import dataclass
from typing import Dict, List

def compute_fft_features(data: np.ndarray, sample_rate: float = 25600.0) -> Dict[str, float]:
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)
    fft_mag = np.abs(np.fft.rfft(data)) / n
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))
    crest_factor = peak / max(rms, 1e-10)
    kurtosis = np.mean(((data - np.mean(data))/max(np.std(data),1e-10))**4)
    dominant_idx = np.argmax(fft_mag[1:]) + 1
    dominant_freq = freqs[dominant_idx]
    return {
        "rms": rms, "peak": peak, "crest_factor": crest_factor,
        "kurtosis": kurtosis, "dominant_freq_hz": dominant_freq,
    }

def bearing_fault_frequencies(rpm: float, n_balls: int = 8,
                               ball_diam: float = 0.5, pitch_diam: float = 2.0) -> Dict[str, float]:
    shaft_hz = rpm / 60
    bd_pd = ball_diam / pitch_diam
    bpfo = n_balls / 2 * shaft_hz * (1 - bd_pd)
    bpfi = n_balls / 2 * shaft_hz * (1 + bd_pd)
    return {"BPFO": bpfo, "BPFI": bpfi, "1x": shaft_hz, "2x": 2*shaft_hz}

if __name__ == "__main__":
    np.random.seed(42)
    SR = 25600.0
    RPM = 1800
    t = np.linspace(0, 1.0, int(SR))
    fault_freqs = bearing_fault_frequencies(RPM)
    print("Fault Frequencies:")
    for name, freq in fault_freqs.items():
        print(f"  {name}: {freq:.2f} Hz")

    healthy = 0.01 * np.random.randn(len(t)) + 0.005 * np.sin(2*np.pi * RPM/60 * t)
    faulty = healthy.copy()
    faulty += 0.1 * np.sin(2*np.pi * fault_freqs["BPFO"] * t)

    for label, sig in [("Healthy", healthy), ("Faulty", faulty)]:
        feats = compute_fft_features(sig, SR)
        print(f"[{label}] kurtosis={feats['kurtosis']:.3f} rms={feats['rms']:.6f} "
              f"dominant={feats['dominant_freq_hz']:.1f}Hz")
        if feats["kurtosis"] > 3.5:
            print(f"  -> FAULT DETECTED (high kurtosis)")
        else:
            print(f"  -> Normal operation")
