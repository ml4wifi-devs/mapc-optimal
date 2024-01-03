import jax.numpy as jnp
from chex import Array


# TGax channel model
# https://www.ieee802.org/11/Reports/tgax_update.htm#:~:text=TGax%20Selection%20Procedure-,11%2D14%2D0980,-TGax%20Simulation%20Scenarios
DEFAULT_TX_POWER = 16.0206  # (40 mW) https://www.nsnam.org/docs/release/3.40/doxygen/d0/d7d/wifi-phy_8cc_source.html#l00171
NOISE_FLOOR = -93.97        # https://www.nsnam.org/docs/models/html/wifi-testing.html#packet-error-rate-performance
NOISE_FLOOR_LIN = jnp.power(10, NOISE_FLOOR / 10)
DEFAULT_SIGMA = 2.          # https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=908165
BREAKING_POINT = 10.        # https://mentor.ieee.org/802.11/dcn/14/11-14-0980-16-00ax-simulation-scenarios.docx (p. 19)
CENTRAL_FREQUENCY = 5.160   # (GHz) https://en.wikipedia.org/wiki/List_of_WLAN_channels#5_GHz_(802.11a/h/n/ac/ax)
REFERENCE_DISTANCE = 1.0
WALL_LOSS = 7.


def path_loss(distance: Array, walls: Array) -> Array:
    return (40.05 + 20 * jnp.log10((jnp.minimum(distance, BREAKING_POINT) * CENTRAL_FREQUENCY) / 2.4) +
            (distance > BREAKING_POINT) * 35 * jnp.log10(distance / BREAKING_POINT) + WALL_LOSS * walls)


# Data rates for IEEE 802.11ax standard, 20 MHz channel width, 1 spatial stream, and 800 ns GI
DATA_RATES = jnp.array([8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.2])

MAX_TX_POWER = 20.  # (dBm)
MIN_TX_POWER = 0.   # (dBm)

# Tx slot duration
TAU = 5.484 * 1e-3          # (s) https://ieeexplore.ieee.org/document/8930559
FRAME_LEN = jnp.asarray(1500 * 8)

# Parameters of the success probability curves - mean of the normal distribution with standard deviation of 2
# (derived from ns-3 simulations)
MEAN_SNRS = jnp.array([
    10.613624240405125, 10.647249582547907, 10.660723984151614, 10.682584060100158,
    11.151267538857537, 15.413200906170632, 16.735812667249125, 18.091175930406580,
    21.806290592040960, 23.331824973610920, 29.788906076547470, 31.750234694079595
])
