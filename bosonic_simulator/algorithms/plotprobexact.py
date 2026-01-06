import numpy as np
import matplotlib.pyplot as plt
from bosonic_simulator.algorithms.measureprobexact import measureprobexact
from bosonic_simulator.gaussian_state_description import GaussianStateDescription


def plotprobexact(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    mode_index: int,
    re_lim: tuple[int, int] | None = None,  # Plot limits in phase space
    im_lim: tuple[int, int] | None = None,
    resolution: int = 15,  # Number of grid points per axis
    cmap: str = "Blues",  # Matplotlib colormap
):

    if re_lim is None:
        variances = [
            psi.covariance_matrix[2 * mode_index, 2 * mode_index]
            for (_, psi) in superposition_terms
        ]
        max_std_dev = max(np.sqrt(v) for v in variances)

        q_centers = [psi.amplitude[mode_index].real for (_, psi) in superposition_terms]

        re_lim = (
            min(q_centers) - 2.0 * max_std_dev,
            max(q_centers) + 2.0 * max_std_dev,
        )

    if im_lim is None:
        variances = [
            psi.covariance_matrix[2 * mode_index + 1, 2 * mode_index + 1]
            for (_, psi) in superposition_terms
        ]
        max_std_dev = max(np.sqrt(v) for v in variances)

        p_centers = [psi.amplitude[mode_index].imag for (_, psi) in superposition_terms]

        im_lim = (
            min(p_centers) - 2.0 * max_std_dev,
            max(p_centers) + 2.0 * max_std_dev,
        )

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

            from bosonic_simulator.algorithms.measureprobapproximate import (
                measureprobapproximate,
            )

            p_approx = measureprobapproximate(
                superposition_terms,
                np.array([Re[i, j] + 1j * Im[i, j]]),
                wires=[mode_index],
                multiplicative_error=0.75,
                max_failure_probability=0.25,
                energy_upper_bound=10,
            )

            prob[i, j] = np.real((p_approx - p) / p)

            print(prob[i, j])

            # from bosonic_simulator.algorithms.postmeasure import postmeasure
            # import bosonic_simulator.algorithms.prob as prr

            # amplitude = np.array([Re[i, j] + 1j * Im[i, j]])
            # pi_alpha_psi = []
            # for c_j, psi_j in superposition_terms:

            #     pi_alpha_psi_j_norm_squared = (np.pi**1) * prr.prob(
            #         psi_j, amplitude, [mode_index]
            #     )
            #     c_prime_j = c_j * np.sqrt(pi_alpha_psi_j_norm_squared)

            #     psi_prime_j = postmeasure(psi_j, amplitude, [mode_index])

            #     pi_alpha_psi.append((c_prime_j, psi_prime_j))

            # prob[i, j] = np.log10(
            #     np.square(
            #         np.sum(
            #             [np.abs(c) * np.sqrt(psi.energy()) for (c, psi) in pi_alpha_psi]
            #         )
            #     )
            #     / (exactnorm(pi_alpha_psi) ** 2)
            # )

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
    plt.title(f"Probability density (mode_index = {mode_index})")
    plt.tight_layout()
    plt.show()

    return re, im, prob


alpha = 1 + 1j
mode = 0


from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.applyunitary import applyunitary
from bosonic_simulator.gaussian_unitary_description import (
    DisplacementDescription,
    SqueezingDescription,
    PhaseShiftDescription,
)

# example cat state |α> + |−α>
superposition_terms = [
    (
        1 / np.sqrt(2),
        applyunitary(
            GaussianStateDescription.vacuum_state(1),
            DisplacementDescription(np.array([alpha])),
        ),
    ),
    (
        1 / np.sqrt(2),
        applyunitary(
            GaussianStateDescription.vacuum_state(1),
            DisplacementDescription(np.array([-alpha])),
        ),
    ),
    (
        1 / np.sqrt(2),
        applyunitary(
            GaussianStateDescription.vacuum_state(1),
            DisplacementDescription(np.array([1 - 1j])),
        ),
    ),
]

from bosonic_simulator.algorithms.exactnorm import exactnorm
from bosonic_simulator.algorithms.fastnorm import fastnorm

exnorm11 = exactnorm(superposition_terms)
superposition_terms = [(c / exnorm11, psi) for (c, psi) in superposition_terms]

energy_upper_bound = np.square(
    np.sum([np.abs(c) * np.sqrt(psi.energy()) for (c, psi) in superposition_terms])
)  # bound energy of superposition state with Cauchy-Schwarz
print(f"energy bound: {energy_upper_bound}")
# negative_count = 0
# inacurrate_count = 0
# n_iterations = 100
# rel_error = 0.4
# rel_error_sum = 0
# for _ in range(n_iterations):
#     exnorm = exactnorm(superposition_terms=superposition_terms)
#     fnorm = fastnorm(superposition_terms, rel_error, 0.1, 10)

#     # print(exnorm)
#     # print(fnorm)
#     print(f"rel = {(fnorm-exnorm) / exnorm}")
#     rel_error_sum += (fnorm - exnorm) / exnorm
#     if fnorm < exnorm:
#         negative_count += 1
#     if np.abs(fnorm - exnorm) / exnorm > rel_error:
#         inacurrate_count += 1

# print(f"negative number proportion: {negative_count/n_iterations}")
# print(f"inacurrate number proportion: {inacurrate_count/n_iterations}")
# print(f"avg. rel. error: {rel_error_sum/n_iterations}")

# superposition_terms = [
#     (c, applyunitary(psi, PhaseShiftDescription(np.float64(-np.pi / 4), 0)))
#     for (c, psi) in superposition_terms
# ]

# superposition_terms = [
#     (c, applyunitary(psi, DisplacementDescription(np.array([1 + 1j]))))
#     for (c, psi) in superposition_terms
# ]

# superposition_terms = [
#     (c, applyunitary(psi, SqueezingDescription(np.log(2), 0)))
#     for (c, psi) in superposition_terms
# ]

plotprobexact(
    superposition_terms=superposition_terms,
    mode_index=0,
    resolution=8,
    cmap="Blues",
    re_lim=(-2, 2),
    im_lim=(-2, 2),
)

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
#     [np.repeat([1.35 * np.exp(1j * np.pi * (2 / 3))], 2), np.array([1 - 0.5j])]
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
#             applyunitary(coh_state, gud.SqueezingDescription(0.2, 1)),
#             gud.DisplacementDescription(np.repeat([-2], 3)),
#         ),
#     ),
#     (1, applyunitary(coh_state, gud.DisplacementDescription(np.repeat([-1], 3)))),
# ]

# superpos_norm = exactnorm(superposition_terms)

# print(superpos_norm)

# superposition_terms = [(c / superpos_norm, psi) for (c, psi) in superposition_terms]

# print(exactnorm(superposition_terms))

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=20,
# )

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=1,
#     resolution=20,
# )

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=2,
#     resolution=20,
# )

# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.exactnorm import exactnorm
# from bosonic_simulator.algorithms.applyunitary import applyunitary
# import bosonic_simulator.gaussian_unitary_description as gud

# alpha = np.concatenate(
#     [np.repeat([1.35 * np.exp(1j * np.pi * (2 / 3))], 2), np.array([1 - 0.5j])]
# )

# coh_state = GaussianStateDescription.coherent_state(np.array(alpha))

# superposition_terms = [
#     (
#         1,
#         applyunitary(
#             GaussianStateDescription.vacuum_state(3),
#             gud.DisplacementDescription(
#                 np.array([-0.674 + 1.169j, -0.674 + 1.169j, -0.674 + 1.169j])
#             ),
#         ),
#     ),
#     (
#         1,
#         applyunitary(
#             GaussianStateDescription.vacuum_state(3),
#             gud.DisplacementDescription(np.array([1 + 0j, 1 + 0j, 1 + 0j])),
#         ),
#     ),
#     (
#         1,
#         applyunitary(
#             GaussianStateDescription.vacuum_state(3),
#             gud.DisplacementDescription(np.array([0 + 1j, 0 + 1.5j, 1.5 + -1j])),
#         ),
#     ),
# ]

# superposition_terms = [
#     (
#         1.0,
#         applyunitary(
#             psi, gud.DisplacementDescription(np.array([-1 + 0j, -1 + 0j, -1 + 0j]))
#         ),
#     )
#     for (_, psi) in superposition_terms
# ]

# superpos_norm = exactnorm(superposition_terms)

# print(superpos_norm)

# superposition_terms = [(c / superpos_norm, psi) for (c, psi) in superposition_terms]

# print(exactnorm(superposition_terms))

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=20,
# )

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=1,
#     resolution=20,
# )

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=2,
#     resolution=20,
# )

# 0.569+0i

# alpha = 1 + 1j
# mode = 0


# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.applyunitary import applyunitary
# from bosonic_simulator.gaussian_unitary_description import (
#     DisplacementDescription,
#     SqueezingDescription,
#     PhaseShiftDescription,
# )

# # example cat state |α> + |−α>
# superposition_terms = [
#     (
#         1 / np.sqrt(2),
#         applyunitary(
#             GaussianStateDescription.vacuum_state(1),
#             DisplacementDescription(np.array([alpha])),
#         ),
#     ),
#     (
#         1 / np.sqrt(2),
#         applyunitary(
#             GaussianStateDescription.vacuum_state(1),
#             DisplacementDescription(np.array([-alpha])),
#         ),
#     ),
# ]

# # superposition_terms = [
# #     (c, applyunitary(psi, PhaseShiftDescription(np.float64(-np.pi / 4), 0)))
# #     for (c, psi) in superposition_terms
# # ]

# superposition_terms = [
#     (c, applyunitary(psi, SqueezingDescription(np.log(2), 0)))
#     for (c, psi) in superposition_terms
# ]

# plotprobexact(
#     superposition_terms=superposition_terms,
#     mode_index=0,
#     resolution=25,
# )

# alpha = 1 + 1j
# mode = 0


# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.applyunitary import applyunitary
# from bosonic_simulator.gaussian_unitary_description import (
#     DisplacementDescription,
#     SqueezingDescription,
#     PhaseShiftDescription,
# )

# # example cat state |α> + |−α>
# superposition_terms = [
#     (
#         1 / np.sqrt(2),
#         applyunitary(
#             GaussianStateDescription.vacuum_state(3),
#             DisplacementDescription(np.array([alpha, alpha, alpha])),
#         ),
#     ),
#     (
#         1 / np.sqrt(2),
#         applyunitary(
#             applyunitary(
#                 GaussianStateDescription.vacuum_state(3),
#                 DisplacementDescription(np.array([-alpha, -alpha, -alpha])),
#             ),
#             SqueezingDescription(0.3, 1),
#         ),
#     ),
#     (
#         1 / np.sqrt(2),
#         applyunitary(
#             applyunitary(
#                 GaussianStateDescription.vacuum_state(3),
#                 DisplacementDescription(np.array([-alpha, -alpha, -alpha])),
#             ),
#             SqueezingDescription(0.3, 1),
#         ),
#     ),
# ]


# from bosonic_simulator.algorithms.exactnorm import exactnorm
# from bosonic_simulator.algorithms.fastnorm import fastnorm

# exnorm1 = exactnorm(superposition_terms)

# superposition_terms = [(c / exnorm1, psi) for (c, psi) in superposition_terms]

# # energy_upper_bound = np.square(
# #     np.sum([np.abs(c) * np.sqrt(psi.energy()) for (c, psi) in superposition_terms])
# # )  # bound energy of superposition state with Cauchy-Schwarz
# # print(f"energy bound: {energy_upper_bound}")
# # negative_count = 0
# # inacurrate_count = 0
# # n_iterations = 100
# # rel_error = 0.4
# # rel_error_sum = 0
# # for _ in range(n_iterations):
# #     exnorm = exactnorm(superposition_terms=superposition_terms)
# #     fnorm = fastnorm(superposition_terms, rel_error, 0.1, 10)

# #     # print(exnorm)
# #     # print(fnorm)
# #     print(f"rel = {(fnorm-exnorm) / exnorm}")
# #     rel_error_sum += (fnorm - exnorm) / exnorm
# #     if fnorm < exnorm:
# #         negative_count += 1
# #     if np.abs(fnorm - exnorm) / exnorm > rel_error:
# #         inacurrate_count += 1

# # print(f"negative number proportion: {negative_count/n_iterations}")
# # print(f"inacurrate number proportion: {inacurrate_count/n_iterations}")
# # print(f"avg. rel. error: {rel_error_sum/n_iterations}")

# # superposition_terms = [
# #     (c, applyunitary(psi, PhaseShiftDescription(np.float64(-np.pi / 4), 0)))
# #     for (c, psi) in superposition_terms
# # ]

# # superposition_terms = [
# #     (c, applyunitary(psi, DisplacementDescription(np.array([1 + 1j]))))
# #     for (c, psi) in superposition_terms
# # ]

# # superposition_terms = [
# #     (c, applyunitary(psi, SqueezingDescription(np.log(2), 0)))
# #     for (c, psi) in superposition_terms
# # ]

# plotprobexact(
#     superposition_terms=superposition_terms, mode_index=0, resolution=8, cmap="Blues"
# )

# plotprobexact(
#     superposition_terms=superposition_terms, mode_index=1, resolution=8, cmap="Blues"
# )

# alpha = 1 + 1j
# mode = 0


# from bosonic_simulator.gaussian_state_description import GaussianStateDescription
# from bosonic_simulator.algorithms.applyunitary import applyunitary
# from bosonic_simulator.gaussian_unitary_description import (
#     DisplacementDescription,
#     SqueezingDescription,
#     PhaseShiftDescription,
# )

# # example cat state |α> + |−α>
# superposition_terms = [
#     (
#         1 / np.sqrt(2),
#         applyunitary(
#             GaussianStateDescription.vacuum_state(2),
#             DisplacementDescription(np.array([alpha, alpha])),
#         ),
#     ),
#     (
#         1 / np.sqrt(2),
#         applyunitary(
#             applyunitary(
#                 GaussianStateDescription.vacuum_state(2),
#                 DisplacementDescription(np.array([-alpha, -alpha])),
#             ),
#             SqueezingDescription(0.3, 1),
#         ),
#     ),
# ]


# from bosonic_simulator.algorithms.exactnorm import exactnorm
# from bosonic_simulator.algorithms.fastnorm import fastnorm

# exnorm1 = exactnorm(superposition_terms)

# superposition_terms = [(c / exnorm1, psi) for (c, psi) in superposition_terms]

# # energy_upper_bound = np.square(
# #     np.sum([np.abs(c) * np.sqrt(psi.energy()) for (c, psi) in superposition_terms])
# # )  # bound energy of superposition state with Cauchy-Schwarz
# # print(f"energy bound: {energy_upper_bound}")
# # negative_count = 0
# # inacurrate_count = 0
# # n_iterations = 100
# # rel_error = 0.4
# # rel_error_sum = 0
# # for _ in range(n_iterations):
# #     exnorm = exactnorm(superposition_terms=superposition_terms)
# #     fnorm = fastnorm(superposition_terms, rel_error, 0.1, 10)

# #     # print(exnorm)
# #     # print(fnorm)
# #     print(f"rel = {(fnorm-exnorm) / exnorm}")
# #     rel_error_sum += (fnorm - exnorm) / exnorm
# #     if fnorm < exnorm:
# #         negative_count += 1
# #     if np.abs(fnorm - exnorm) / exnorm > rel_error:
# #         inacurrate_count += 1

# # print(f"negative number proportion: {negative_count/n_iterations}")
# # print(f"inacurrate number proportion: {inacurrate_count/n_iterations}")
# # print(f"avg. rel. error: {rel_error_sum/n_iterations}")

# # superposition_terms = [
# #     (c, applyunitary(psi, PhaseShiftDescription(np.float64(-np.pi / 4), 0)))
# #     for (c, psi) in superposition_terms
# # ]

# # superposition_terms = [
# #     (c, applyunitary(psi, DisplacementDescription(np.array([1 + 1j]))))
# #     for (c, psi) in superposition_terms
# # ]

# # superposition_terms = [
# #     (c, applyunitary(psi, SqueezingDescription(np.log(2), 0)))
# #     for (c, psi) in superposition_terms
# # ]

# plotprobexact(
#     superposition_terms=superposition_terms, mode_index=0, resolution=8, cmap="Blues"
# )

# plotprobexact(
#     superposition_terms=superposition_terms, mode_index=1, resolution=8, cmap="Blues"
# )
