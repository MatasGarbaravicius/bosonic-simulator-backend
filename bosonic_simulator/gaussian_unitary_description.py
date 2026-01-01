from dataclasses import (
    dataclass,
)
from typing import Union
import numpy as np


@dataclass  # automatically adds generated methods such as __init__()
class SqueezingDescription:
    parameter: np.float64
    mode_index: int


@dataclass
class DisplacementDescription:
    amplitude: np.ndarray


@dataclass
class PhaseShiftDescription:
    angle: np.float64
    mode_index: int


@dataclass
class BeamSplitterDescription:
    angle: np.float64
    mode_j_index: int
    mode_k_index: int


GaussianUnitaryDescription = Union[
    SqueezingDescription,
    DisplacementDescription,
    PhaseShiftDescription,
    BeamSplitterDescription,
]

# b = PhaseShiftDescription(1.0,5)
# print(isinstance(b, GaussianUnitaryDescription)
