import numpy as np
import matplotlib.pyplot as plt
from bosonic_simulator.algorithms.measureprobexact import measureprobexact
from bosonic_simulator.gaussian_state_description import GaussianStateDescription


def plotprobexact(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    mode_index: int,
    re_lim: tuple[int, int] = (-3, 3),  # Plot limits in phase space
    im_lim: tuple[int, int] = (-3, 3),
    resolution: int = 200,  # Number of grid points per axis
    cmap: str = "Blues",  # Matplotlib colormap
):

    # phase-space grid
    re = np.linspace(re_lim[0], re_lim[1], resolution)
    im = np.linspace(im_lim[0], im_lim[1], resolution)
    Re, Im = np.meshgrid(re, im)

    prob = np.zeros_like(Re, dtype=float)

    for i in range(resolution):
        for j in range(resolution):
            p = measureprobexact(
                superposition_terms,
                np.array([Re[i, j] + 1j * Im[i, j]]),
                wires=[mode_index],
            )

            prob[i, j] = np.real(p)

    # plot
    plt.figure(figsize=(6, 5))
    plt.imshow(
        prob,
        extent=[re.min(), re.max(), im.min(), im.max()],
        origin="lower",
        cmap=cmap,
    )
    plt.colorbar(label=r"$p_{\Psi}(\beta)$")
    plt.xlabel(r"$\mathrm{Re}(\beta)$")
    plt.ylabel(r"$\mathrm{Im}(\beta)$")
    plt.title("Probability density")
    plt.tight_layout()
    plt.show()

    return re, im, prob


# alpha = (-1 + 1j) * np.exp(1j * np.pi / 8)
# mode = 0


# from bosonic_simulator.gaussian_state_description import GaussianStateDescription

# # example cat state |α> + |−α>
# superposition_terms = [
#     (1 / np.sqrt(2), GaussianStateDescription.coherent_state(np.array([alpha]))),
#     (1 / np.sqrt(2), GaussianStateDescription.coherent_state(np.array([-alpha]))),
# ]

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=25,
# )

# alpha = (-1 + 1j) * np.exp(1j * np.pi / 8)
# mode = 0


# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.exactnorm import exactnorm

# alpha = 1.35 * np.exp(1j * np.pi * (2 / 3))

# superposition_terms = [
#     (1, GaussianStateDescription.coherent_state(np.array([alpha]))),
#     (1, GaussianStateDescription.coherent_state(np.array([alpha**2]))),
#     (1, GaussianStateDescription.coherent_state(np.array([alpha**3]))),
# ]

# superpos_norm = exactnorm(superposition_terms)

# superposition_terms = [(c / superpos_norm, psi) for (c, psi) in superposition_terms]

# print(exactnorm(superposition_terms))

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=25,
# )

# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.exactnorm import exactnorm
# from bosonic_simulator.algorithms.applyunitary import applyunitary
# import bosonic_simulator.gaussian_unitary_description as gud

# alpha = 1.35 * np.exp(1j * np.pi * (2 / 3))

# superposition_terms = [
#     (1, GaussianStateDescription.coherent_state(np.array([alpha]))),
#     (
#         1,
#         applyunitary(
#             GaussianStateDescription.coherent_state(np.array([alpha])),
#             gud.PhaseShiftDescription(np.pi / 2, 0),
#         ),
#     ),
#     (
#         1,
#         applyunitary(
#             GaussianStateDescription.coherent_state(np.array([alpha])),
#             gud.PhaseShiftDescription(np.pi, 0),
#         ),
#     ),
# ]

# superpos_norm = exactnorm(superposition_terms)

# superposition_terms = [(c / superpos_norm, psi) for (c, psi) in superposition_terms]

# print(exactnorm(superposition_terms))

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=25,
# )

# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.exactnorm import exactnorm
# from bosonic_simulator.algorithms.applyunitary import applyunitary
# import bosonic_simulator.gaussian_unitary_description as gud

# alpha = 1.35 * np.exp(1j * np.pi * (2 / 3))

# superposition_terms = [
#     (1, GaussianStateDescription.coherent_state(np.array([alpha]))),
#     (
#         1,
#         applyunitary(
#             applyunitary(
#                 GaussianStateDescription.coherent_state(np.array([alpha])),
#                 gud.PhaseShiftDescription(np.pi / 2, 0),
#             ),
#             gud.SqueezingDescription(0.2, 0),
#         ),
#     ),
#     (
#         1,
#         applyunitary(
#             applyunitary(
#                 GaussianStateDescription.coherent_state(np.array([alpha])),
#                 gud.PhaseShiftDescription(np.pi, 0),
#             ),
#             gud.SqueezingDescription(0.75, 0),
#         ),
#     ),
# ]

# superpos_norm = exactnorm(superposition_terms)

# superposition_terms = [(c / superpos_norm, psi) for (c, psi) in superposition_terms]

# print(exactnorm(superposition_terms))

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=25,
# )

# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.exactnorm import exactnorm
# from bosonic_simulator.algorithms.applyunitary import applyunitary
# import bosonic_simulator.gaussian_unitary_description as gud

# alpha = 1.35 * np.exp(1j * np.pi * (2 / 3))

# superposition_terms = [
#     (
#         1,
#         applyunitary(
#             applyunitary(
#                 GaussianStateDescription.coherent_state(np.array([alpha])),
#                 gud.SqueezingDescription(5, 0),
#             ),
#             gud.PhaseShiftDescription(np.pi * (1 / 9), 0),
#         ),
#     ),
# ]

# superpos_norm = exactnorm(superposition_terms)

# superposition_terms = [(c / superpos_norm, psi) for (c, psi) in superposition_terms]

# print(exactnorm(superposition_terms))

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=25,
# )

# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.exactnorm import exactnorm
# from bosonic_simulator.algorithms.applyunitary import applyunitary
# import bosonic_simulator.gaussian_unitary_description as gud

# alpha = np.concatenate(
#     [np.repeat([1.35 * np.exp(1j * np.pi * (1 / 3))], 2), np.array([1])]
# )

# coh_state = GaussianStateDescription.coherent_state(np.array(alpha))

# superposition_terms = [
#     (
#         1,
#         applyunitary(
#             applyunitary(coh_state, gud.PhaseShiftDescription(np.pi / 3, 0)),
#             gud.DisplacementDescription(np.repeat([-2], 3)),
#         ),
#     ),
#     (
#         1,
#         applyunitary(
#             applyunitary(coh_state, gud.SqueezingDescription(2, 1)),
#             gud.DisplacementDescription(np.repeat([-2], 3)),
#         ),
#     ),
#     (1, applyunitary(coh_state, gud.DisplacementDescription(np.repeat([-2], 3)))),
# ]

# superpos_norm = exactnorm(superposition_terms)

# superposition_terms = [(c / superpos_norm, psi) for (c, psi) in superposition_terms]

# print(exactnorm(superposition_terms))

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=10,
# )

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=1,
#     resolution=10,
# )

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=2,
#     resolution=10,
# )
