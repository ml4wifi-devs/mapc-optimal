import os
os.environ['JAX_ENABLE_X64'] = 'True'

from mapc_optimal.solver import Solver
from mapc_optimal.utils import positions_to_path_loss
