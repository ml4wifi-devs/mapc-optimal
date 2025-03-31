"""
Default values of the parameters used in the solver.
"""

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

DATA_RATES = {
    20:  np.array([  8.6,  17.2,  25.8,  34.4,  51.6,   68.8,   77.4,   86.0,  103.2,  114.7,  129.0,  143.2,  154.9,  172.1]),
    40:  np.array([ 17.2,  34.4,  51.6,  68.8, 103.2,  137.6,  154.9,  172.1,  206.5,  229.4,  258.1,  286.8,  309.7,  344.1]),
    80:  np.array([ 36.0,  72.1, 108.1, 144.1, 216.2,  288.2,  324.3,  360.3,  432.4,  480.4,  540.4,  600.5,  648.5,  720.6]),
    160: np.array([ 72.1, 144.1, 216.2, 288.2, 432.4,  576.5,  648.5,  720.6,  864.7,  960.8, 1080.9, 1201.0, 1297.1, 1441.2])
}
r"""Data rates for IEEE 802.11be standard, 1 spatial stream, and 800 ns GI (Mb/s)"""

MEAN_SNRS = {
    20:  np.array([15.160, 13.720, 12.749, 12.315, 11.816, 13.850, 14.639, 15.660, 19.442, 20.892, 28.141, 30.084, 33.888, 35.913]),
    40:  np.array([13.937, 12.314, 11.807, 11.671, 12.610, 15.901, 17.166, 18.447, 22.386, 23.885, 31.153, 33.082, 36.892, 38.926]),
    80:  np.array([12.287, 11.475, 11.209, 12.432, 14.802, 18.870, 20.203, 21.485, 25.403, 26.908, 34.376, 36.301, 40.107, 42.129]),
    160: np.array([11.492, 11.342, 12.263, 14.681, 17.739, 21.901, 23.215, 24.481, 28.421, 29.906, 37.386, 39.310, 43.131, 45.163])
}
r"""Parameters of the success probability curves - cdf of the normal distribution with standard deviation of 1.6
(derived from ns-3 simulations with Nakagami fading)"""

MIN_SNRS = {channel_width: MEAN_SNRS[channel_width] + 2.6317658 for channel_width in MEAN_SNRS}
r"""Minimal SNR values for each MCS (dB) to obtain 95% success probability"""
