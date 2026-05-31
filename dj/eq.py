"""Three-band DJ EQ (low / mid / high) built on scipy biquads.

Used by the mixer for bass-swap crossfades and for in-track EQ kills. The bands
are defined by *subtraction* — low is a lowpass, high is a highpass, and mid is
whatever's left (input - low - high). So at unity gain the three bands sum back
to the input exactly: the EQ is transparent until a band is actually cut, and
the mids never need their own filter. Filter state is carried across blocks so
there are no clicks at block boundaries or when a band gain changes mid-stream.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from . import CHANNELS, SAMPLE_RATE


class ThreeBandEQ:
    def __init__(
        self,
        low_hz: float = 200.0,
        high_hz: float = 2000.0,
        order: int = 4,
        sr: int = SAMPLE_RATE,
        channels: int = CHANNELS,
    ):
        self.channels = channels
        nyq = sr / 2
        self._sos_low = butter(order, low_hz / nyq, btype="low", output="sos")
        self._sos_high = butter(order, high_hz / nyq, btype="high", output="sos")
        self._zi_low = self._fresh_zi(self._sos_low)
        self._zi_high = self._fresh_zi(self._sos_high)

    def _fresh_zi(self, sos: np.ndarray) -> np.ndarray:
        return np.zeros((sos.shape[0], 2, self.channels), dtype=np.float64)

    def reset(self) -> None:
        """Clear filter memory so a previous track's tail doesn't bleed in."""
        self._zi_low = self._fresh_zi(self._sos_low)
        self._zi_high = self._fresh_zi(self._sos_high)

    def process(self, block: np.ndarray, low_g: float, mid_g: float, high_g: float) -> np.ndarray:
        low, self._zi_low = sosfilt(self._sos_low, block, axis=0, zi=self._zi_low)
        high, self._zi_high = sosfilt(self._sos_high, block, axis=0, zi=self._zi_high)
        mid = block - low - high
        return (low_g * low + mid_g * mid + high_g * high).astype(np.float32)
