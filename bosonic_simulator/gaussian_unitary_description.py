from dataclasses import (
    dataclass,
)
from typing import Union
import numpy as np
from numpy.typing import NDArray


@dataclass
class SqueezingDescription:
    r"""
    Description of a single-mode squeezing operation $S_j(z)$.

    This class captures the parameters for the squeezing operator described in
    Section 3.3 of the reference paper. The operator is associated with:

    * A squeezing parameter $z \in (0, +\infty)$.
    * A target mode $j$.

    Parameters
    ----------
    parameter : np.float64
        The squeezing parameter $z$.
    mode_index : int
        The index of the mode $j$ to be squeezed (0-based indexing).
    """

    parameter: np.float64
    mode_index: int


@dataclass
class DisplacementDescription:
    r"""
    Description of an n-mode displacement operation $D(\alpha)$.

    This class captures the parameters for the displacement operator described in
    Section 3.3 of the reference paper. The operator is determined by
    a complex vector $\alpha \in \mathbb{C}^n$.

    Parameters
    ----------
    amplitude : NDArray[np.complex128]
        The complex displacement vector $\alpha$.
    """

    amplitude: NDArray[np.complex128]


@dataclass
class PhaseShiftDescription:
    r"""
    Description of a single-mode phase shift operation $F_j(\phi)$.

    This class captures the parameters for the phase shift operator described in
    Section 3.3 of the reference paper. The operator is associated with:

    * A rotation angle $\phi \in \mathbb{R}$.
    * A target mode $j$.

    Parameters
    ----------
    angle : np.float64
        The phase shift angle $\phi$.
    mode_index : int
        The index of the mode $j$ to which the phase shift is applied (0-based indexing).
    """

    angle: np.float64
    mode_index: int


@dataclass
class BeamSplitterDescription:
    r"""
    Description of a beam splitter operation $B_{j,k}(\omega)$.

    This class captures the parameters for the beam splitter operator described in
    Section 3.3 of the reference paper. The operator is associated with:

    * A rotation angle $\omega \in \mathbb{R}$.
    * Two distinct modes $j$ and $k$.

    Parameters
    ----------
    angle : np.float64
        The beam splitter angle $\omega$.
    mode_j_index : int
        The index of the first mode $j$ (0-based indexing).
    mode_k_index : int
        The index of the second mode $k$ (0-based indexing).
    """

    angle: np.float64
    mode_j_index: int
    mode_k_index: int


GaussianUnitaryDescription = Union[
    SqueezingDescription,
    DisplacementDescription,
    PhaseShiftDescription,
    BeamSplitterDescription,
]
r"""
Type alias for any valid Gaussian unitary description supported by the simulator.
"""

# b = PhaseShiftDescription(1.0,5)
# print(isinstance(b, GaussianUnitaryDescription)
