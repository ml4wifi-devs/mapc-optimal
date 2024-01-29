import numpy as np


MAX_TX_POWER = 20.
r"""Physical constant (dBm)"""

MIN_TX_POWER = 10.
r"""Physical constant (dBm)"""

NOISE_FLOOR = -93.97
r"""Physical constant (dBm)  
https://www.nsnam.org/docs/models/html/wifi-testing.html#packet-error-rate-performance"""

CENTRAL_FREQUENCY = 5.160
r"""Simulation parameter (GHz) 
https://en.wikipedia.org/wiki/List_of_WLAN_channels#5_GHz_(802.11a/h/n/ac/ax)"""

WALL_LOSS = 7.
r"""Simulation parameter (dB)"""

BREAKING_POINT = 10.
r"""TGax channel model parameter (m) 
https://mentor.ieee.org/802.11/dcn/14/11-14-0980-16-00ax-simulation-scenarios.docx (p. 19)"""

REFERENCE_DISTANCE = 1.
r"""TGax channel model parameter (m)"""

DATA_RATES = np.array([8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.2])
r"""Data rates for IEEE 802.11ax standard, 20 MHz channel width, 1 spatial stream, and 800 ns GI (Mb/s)"""

MEAN_SNRS = np.array([
    10.613624240405125, 10.647249582547907, 10.660723984151614, 10.682584060100158,
    11.151267538857537, 15.413200906170632, 16.735812667249125, 18.091175930406580,
    21.806290592040960, 23.331824973610920, 29.788906076547470, 31.750234694079595
])
r"""Parameters of the success probability curves - cdf of the normal distribution with standard deviation of 2
(derived from ns-3 simulations)"""

MIN_SNRS = MEAN_SNRS + 5.1516595
r"""Minimal SNR values for each MCS (dB) to obtain 99.5% success probability"""
