# xdsp_filters_bessel_eq
```python
#!/usr/bin/env python3
"""
xdsp_bessel_eq.py
-----------------
High-order parametric EQ built from Bessel prototype Qs.

This produces a smoother, more time-coherent “analog mastering”
feel than Butterworth EQs — less steep, more transient-friendly.

Implements:
    • rbj_peak()   – standard RBJ peaking biquad
    • bessel_Qs()  – Q distribution from Bessel poles
    • design_bessel_peak() – cascaded high-order PEQ
    • cascade_freq_response() – response calculator

Author: James Theory / GPT-5
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt


# =========================================================
# Data structure
# =========================================================

@dataclass
class Biquad:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


# =========================================================
# RBJ Peaking Filter
# =========================================================

def rbj_peak(f0: float, fs: float, Q: float, gain_db: float) -> Biquad:
    """Classic RBJ peaking EQ biquad."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * pi * (f0 / fs)
    alpha = sin(w0) / (2.0 * Q)
    cw = cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A

    # normalize
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return Biquad(b0, b1, b2, a1, a2)


# =========================================================
# Bessel Prototype Qs
# =========================================================

def bessel_Qs(order: int) -> List[float]:
    """
    Return per-section Q values derived from normalized Bessel polynomials
    (maximally flat group delay). Tabulated from standard pole locations.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("order must be even and >= 2")

    # ζ = damping = 1/(2Q)
    bessel_zeta = {
        2: [0.57735],
        4: [0.5219, 0.8055],
        6: [0.5120, 0.6617, 0.8800],
        8: [0.5070, 0.6013, 0.7460, 0.9130],
        10: [0.5040, 0.5670, 0.6700, 0.8010, 0.9490],
        12: [0.5020, 0.5450, 0.6240, 0.7250, 0.8470, 0.9770],
    }
    if order not in bessel_zeta:
        raise ValueError("Bessel Qs tabulated for even orders 2–12 only.")

    return [1.0 / (2.0 * ζ) for ζ in bessel_zeta[order]]


# =========================================================
# High-Order Bessel-EQ Design
# =========================================================

def design_bessel_peak(order: int, fs: float, f0: float, gain_db: float) -> List[Biquad]:
    """
    Build a high-order parametric EQ using Bessel prototype Qs.

    - order: even total order (2, 4, 6, 8, ...)
    - fs: sampling rate (Hz)
    - f0: center frequency (Hz)
    - gain_db: total desired gain at f0 (dB)
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("order must be even and >= 2")
    if not (0 < f0 < fs * 0.5):
        raise ValueError("f0 must be between 0 and Nyquist.")

    Qs = bessel_Qs(order)
    n_sections = len(Qs)
    per_stage_gain_db = gain_db / n_sections

    biquads = [rbj_peak(f0=f0, fs=fs, Q=Q, gain_db=per_stage_gain_db) for Q in Qs]
    return biquads


# =========================================================
# Frequency Response
# =========================================================

def cascade_freq_response(biquads: List[Biquad], fs: float, n_fft: int = 4096):
    """Compute frequency response for a cascade of biquads."""
    w = np.linspace(0.0, pi, n_fft)
    z = np.exp(1j * w)
    H = np.ones_like(z, dtype=complex)
    for bq in biquads:
        H *= (bq.b0 + bq.b1 / z + bq.b2 / (z**2)) / (1 + bq.a1 / z + bq.a2 / (z**2))
    f = w * fs / (2.0 * pi)
    return f, H


# =========================================================
# Demo
# =========================================================

if __name__ == "__main__":
    fs = 48000.0
    f0 = 2000.0
    gain_db = 6.0

    orders = [2, 4, 8]
    plt.figure(figsize=(8, 4))

    for order in orders:
        peq = design_bessel_peak(order, fs, f0, gain_db)
        f, H = cascade_freq_response(peq, fs)
        mag = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
        plt.semilogx(f, mag, label=f"Bessel PEQ N={order}")

    plt.axvline(f0, color="gray", ls="--", alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(which="both", alpha=0.3)
    plt.title(f"Bessel High-Order Parametric EQ (+{gain_db} dB @ {int(f0)} Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

```
