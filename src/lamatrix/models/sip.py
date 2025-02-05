"""Implements a special case of a lnGaussian2D that fits for SIP polynomials"""

from typing import List

import numpy as np

from ..distributions import Distribution, DistributionsContainer
from ..io import IOMixins, LatexMixins
from ..math import MathMixins
from ..model import Model
from lamatrix import Polynomial, Constant

__all__ = [
    "SIP",
]


class SIP(MathMixins, Model):
    """Special case of a lnGaussian2D which is designed to fit for SIP coefficients"""

    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        dx_name: str = "dx",
        dy_name: str = "dy",
        order: int = 1,
        priors=None,
        prior_A=None,
        prior_sigma_x=None,
        prior_sigma_y=None,
        prior_mu_x=None,
        prior_mu_y=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self.dx_name = dx_name
        self.dy_name = dy_name
        self.order = order
        self._validate_arg_names()

        super().__init__(priors=priors)
        if np.any(
            [
                (p is not None)
                for p in [prior_A, prior_sigma_x, prior_sigma_y, prior_mu_x, prior_mu_y]
            ]
        ):
            if priors is not None:
                raise ValueError(
                    "Specify either priors on sigma or priors on coefficients."
                )
            prior_sigma_x = self._validate_distributions(prior_sigma_x)[0]
            prior_mu_x = self._validate_distributions(prior_mu_x)[0]
            prior_sigma_y = self._validate_distributions(prior_sigma_y)[0]
            prior_mu_y = self._validate_distributions(prior_mu_y)[0]
            prior_A = self._validate_distributions(prior_A)[0]
            self.priors = self.gaussian_parameters_to_coefficients(
                DistributionsContainer(
                    [prior_A, prior_sigma_x, prior_sigma_y, prior_mu_x, prior_mu_y]
                )
            )

    @property
    def width(self):
        nsquare = 2 * (
            (
                np.diag(np.ones((self.order + 1), bool))
                + np.diag(np.ones((self.order + 1), bool), -1)[1:, 1:]
            ).sum()
            - 1
        ) + (self.order**2)
        return 3 + (2 * (self.order + 1) ** 2) + nsquare

    @property
    def nvectors(self):
        return 4

    @property
    def arg_names(self):
        return {self.x_name, self.y_name, self.dx_name, self.dy_name}

    @property
    def _initialization_attributes(self):
        return [
            "x_name",
            "y_name",
            "dx_name",
            "dy_name",
            "order",
        ]

    def gaussian_parameters_to_coefficients(self, distributions):
        P_width = (self.order + 1) ** 2
        A = distributions[0]
        if isinstance(A, tuple):
            A, A_err = A
        sigma_x = distributions[1]
        if isinstance(sigma_x, tuple):
            sigma_x, sigma_x_err = sigma_x
        sigma_y = distributions[2]
        if isinstance(sigma_y, tuple):
            sigma_y, sigma_y_err = sigma_y

        mu_x = distributions[3 : 3 + P_width]
        if isinstance(mu_x[0], tuple):
            mu_x, mu_x_err = (
                DistributionsContainer(mu_x).mean,
                DistributionsContainer(mu_x).std,
            )
        else:
            mu_x = np.asarray(mu_x)
        mu_y = distributions[3 + P_width : 3 + P_width * 2]
        if isinstance(mu_y[0], tuple):
            mu_y, mu_y_err = (
                DistributionsContainer(mu_y).mean,
                DistributionsContainer(mu_y).std,
            )
        else:
            mu_y = np.asarray(mu_y)

        if (sigma_x <= 0) | (sigma_y <= 0):
            raise ValueError("Invalid input: 'sigma' must be positive.")
        a_x = -1 / (2 * sigma_x**2)
        b_x = mu_x / sigma_x**2
        a_y = -1 / (2 * sigma_y**2)
        b_y = mu_y / sigma_y**2
        c = (
            np.log(A)
            - np.log(2 * np.pi * sigma_x * sigma_y)
            - (mu_x[0] ** 2) / (2 * sigma_x**2)
            - (mu_y[0] ** 2) / (2 * sigma_y**2)
        )
        if isinstance(distributions[0], (int, float)):
            return [a_x, a_y, b_x, b_y, c]
        elif isinstance(distributions[0], tuple):
            a_x_err = (1 / sigma_x**3) * sigma_x_err
            b_x_err = np.sqrt(
                (1 / sigma_x**2 * mu_x_err) ** 2
                + (-2 * mu_x / sigma_x**3 * sigma_x_err) ** 2
            )
            a_y_err = (1 / sigma_y**3) * sigma_y_err
            b_y_err = np.sqrt(
                (1 / sigma_y**2 * mu_y_err) ** 2
                + (-2 * mu_y / sigma_y**3 * sigma_y_err) ** 2
            )
            dc_dA = 1 / A
            dc_dsigma_x = -1 / sigma_x - mu_x[0] ** 2 / sigma_x**3
            dc_dsigma_y = -1 / sigma_y - mu_y[0] ** 2 / sigma_y**3
            dc_dmu_x = -mu_x[0] / sigma_x**2
            dc_dmu_y = -mu_y[0] / sigma_y**2
            c_err = np.sqrt(
                (dc_dA * A_err) ** 2
                + (dc_dsigma_x * sigma_x_err) ** 2
                + (dc_dmu_x * mu_x_err[0]) ** 2
                + (dc_dsigma_y * sigma_y_err) ** 2
                + (dc_dmu_y * mu_y_err[0]) ** 2
            )
            return DistributionsContainer(
                [
                    (a_x, a_x_err),
                    (a_y, a_y_err),
                    *[(b_x0, b_x_err0) for b_x0, b_x_err0 in zip(b_x, b_x_err)],
                    *[(b_y0, b_y_err0) for b_y0, b_y_err0 in zip(b_y, b_y_err)],
                    (c, c_err),
                ]
            )

    def coefficients_to_gaussian_parameters(self, distributions):
        P_width = (self.order + 1) ** 2
        a_x = distributions[0]
        if isinstance(a_x, tuple):
            a_x, a_x_err = a_x
        a_y = distributions[1]
        if isinstance(a_y, tuple):
            a_y, a_y_err = a_y
        b_x = distributions[2 : 2 + P_width]
        if isinstance(b_x[0], tuple):
            b_x, b_x_err = (
                DistributionsContainer(b_x).mean,
                DistributionsContainer(b_x).std,
            )
        else:
            b_x = np.asarray(b_x)
        b_y = distributions[2 + P_width : 2 + P_width * 2]
        if isinstance(b_y[0], tuple):
            b_y, b_y_err = (
                DistributionsContainer(b_y).mean,
                DistributionsContainer(b_y).std,
            )
        else:
            b_y = np.asarray(b_y)
        c = distributions[-1]
        if isinstance(c, tuple):
            c, c_err = c
        if (a_x >= 0) | (a_y >= 0):
            raise ValueError("Invalid input: 'a' must be positive.")

        sigma_x = np.sqrt(-1 / (2 * a_x))
        mu_x = -b_x / (2 * a_x)
        sigma_y = np.sqrt(-1 / (2 * a_y))
        mu_y = -b_y / (2 * a_y)
        A = np.exp(
            c
            + np.log(2 * np.pi * sigma_x * sigma_y)
            + (mu_x[0] ** 2) / (2 * sigma_x**2)
            + (mu_y[0] ** 2) / (2 * sigma_y**2)
        )

        if isinstance(distributions[0], (int, float)):
            return A, sigma_x, sigma_y, mu_x, mu_y

        elif isinstance(distributions[0], tuple):
            sigma_x_err = (1 / (4 * sigma_x * a_x**2)) * a_x_err
            mu_x_err = np.sqrt(
                (b_x / (2 * a_x**2) * a_x_err) ** 2 + (-1 / (2 * a_x) * b_x_err) ** 2
            )

            sigma_y_err = (1 / (4 * sigma_y * a_y**2)) * a_y_err
            mu_y_err = np.sqrt(
                (b_y / (2 * a_y**2) * a_y_err) ** 2 + (-1 / (2 * a_y) * b_y_err) ** 2
            )

            dA_dc = A
            dA_dsigma_x = A * (1 / sigma_x + mu_x[0] ** 2 / sigma_x**3)
            dA_dmu_x = A * (mu_x[0] / sigma_x**2)
            dA_dsigma_y = A * (1 / sigma_y + mu_y[0] ** 2 / sigma_y**3)
            dA_dmu_y = A * (mu_y[0] / sigma_y**2)

            A_err = np.sqrt(
                (dA_dc * c_err) ** 2
                + (dA_dsigma_x * sigma_x_err) ** 2
                + (dA_dmu_x * mu_x_err[0]) ** 2
                + (dA_dsigma_y * sigma_y_err) ** 2
                + (dA_dmu_y * mu_y_err[0]) ** 2
            )
            return DistributionsContainer(
                [
                    (A, A_err),
                    (sigma_x, sigma_x_err),
                    (sigma_y, sigma_y_err),
                    *[(mu_x0, mu_x_err0) for mu_x0, mu_x_err0 in zip(mu_x, mu_x_err)],
                    *[(mu_y0, mu_y_err0) for mu_y0, mu_y_err0 in zip(mu_y, mu_y_err)],
                ]
            )

    def design_matrix(self, **kwargs):
        """Build a design matrix for SIP model.

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """

        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        y = kwargs.get(self.y_name)
        dx = kwargs.get(self.dx_name)
        dy = kwargs.get(self.dy_name)

        P, P2 = get_sip_matrices(x, y, order=self.order)
        ndim = x.ndim
        shape_a = [*np.arange(1, ndim + 1).astype(int), 0]
        shape_b = [ndim, *np.arange(0, ndim)]
        X = np.vstack(
            [
                np.expand_dims(dx, axis=ndim).transpose(shape_b) ** 2,
                np.expand_dims(dy, axis=ndim).transpose(shape_b) ** 2,
                np.expand_dims(dx, axis=ndim).transpose(shape_b) * P.transpose(shape_b),
                np.expand_dims(dy, axis=ndim).transpose(shape_b) * P.transpose(shape_b),
                P2.transpose(shape_b),
                np.ones((*x.shape, 1)).transpose(shape_b),
            ]
        ).transpose(shape_a)
        return X

    @property
    def _equation(self):
        P_str = [
            f"\\mathbf{{{self.latex_aliases[self.y_name]}}}^{idx}"
            + f"\\mathbf{{{self.latex_aliases[self.x_name]}}}^{jdx}"
            for idx in range(self.order + 1)
            for jdx in range(self.order + 1)
        ]

        return [
            f"\\mathbf{{{self.latex_aliases[self.dx_name]}}}^2",
            f"\\mathbf{{{self.latex_aliases[self.dy_name]}}}^2",
            *[
                f"\\mathbf{{{self.latex_aliases[self.dx_name]}}} . (" + s + ")"
                for s in P_str
            ],
            *[
                f"\\mathbf{{{self.latex_aliases[self.dy_name]}}} . (" + s + ")" + s
                for s in P_str
            ],
            "",
        ]

    @property
    def sigma_x(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[1]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[1]
        )

    @property
    def mu_x(self):
        P_width = (self.order + 1) ** 2
        return DistributionsContainer(
            self.coefficients_to_gaussian_parameters(self.posteriors)[3 : 3 + P_width]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[3 : 3 + P_width]
        )

    @property
    def sigma_y(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[2]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[2]
        )

    @property
    def mu_y(self):
        P_width = (self.order + 1) ** 2
        return DistributionsContainer(
            self.coefficients_to_gaussian_parameters(self.posteriors)[
                3 + P_width : 3 + 2 * P_width
            ]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[
                3 + P_width : 3 + 2 * P_width
            ]
        )

    @property
    def A(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[0]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[0]
        )

    def mu_x_to_Polynomial(self):
        """Convert the best fit mu_x values to an lamatrix.model.Polynomial object"""
        poly = (Constant() + Polynomial(self.x_name, order=self.order)) * (
            Constant() + Polynomial(self.y_name, order=self.order)
        )
        mean = self.mu_x.mean.reshape((self.order + 1, self.order + 1))
        std = self.mu_x.std.reshape((self.order + 1, self.order + 1))
        mean, std = (
            np.hstack([mean[0, 0], mean[0, 1:], mean[1:, 0], mean[1:, 1:].ravel()]),
            np.hstack([std[0, 0], std[0, 1:], std[1:, 0], std[1:, 1:].ravel()]),
        )
        poly.posteriors = DistributionsContainer([(m, s) for m, s in zip(mean, std)])
        return poly

    def mu_y_to_Polynomial(self):
        """Convert the best fit mu_x values to an lamatrix.model.Polynomial object"""
        poly = (Constant() + Polynomial(self.x_name, order=self.order)) * (
            Constant() + Polynomial(self.y_name, order=self.order)
        )
        mean = self.mu_y.mean.reshape((self.order + 1, self.order + 1))
        std = self.mu_y.std.reshape((self.order + 1, self.order + 1))
        mean, std = (
            np.hstack([mean[0, 0], mean[0, 1:], mean[1:, 0], mean[1:, 1:].ravel()]),
            np.hstack([std[0, 0], std[0, 1:], std[1:, 0], std[1:, 1:].ravel()]),
        )
        poly.posteriors = DistributionsContainer([(m, s) for m, s in zip(mean, std)])
        return poly


def get_sip_matrices(x, y, order=1):
    """Given an input x and y position as nd arrays, will calculate the matrixes that represent a simple polynomial"""
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    # This function will work with any dimension of input
    ndim = x.ndim
    # transpose shapes
    shape_a = [*np.arange(1, ndim + 1).astype(int), 0]
    shape_b = [ndim, *np.arange(0, ndim)]

    # Polynomial in x
    Px = np.asarray([x**idx for idx in range(order + 1)]).transpose(shape_a)
    # Polynomial in yumn
    Py = np.asarray([y**idx for idx in range(order + 1)]).transpose(shape_a)

    # 2D polynomial
    P = [np.expand_dims(p, axis=ndim) * Py for p in Px.transpose(shape_b)]
    # Reshape such that the last dimension steps through the polynomial orders.
    # Last dimension will have shape (order + 1)**2
    P = np.vstack([p.transpose(shape_b) for p in P]).transpose(shape_a)

    # Build x polynomial squared
    Px2 = [np.expand_dims(p, axis=ndim) * Px for p in Px.transpose(shape_b)]
    # Set last dimension
    Px2 = np.vstack([p.transpose(shape_b) for p in Px2]).transpose(shape_a)
    # Mask out terms which are duplicative
    mask = (
        np.diag(np.ones(Px.shape[-1], bool))
        + np.diag(np.ones(Px.shape[-1], bool), -1)[1:, 1:]
    )
    Px2 = Px2.transpose(shape_b)[mask.ravel()].transpose(shape_a)

    # Build y polynomial squared
    Py2 = [np.expand_dims(p, axis=ndim) * Py for p in Py.transpose(shape_b)]
    # Set last dimension
    Py2 = np.vstack([p.transpose(shape_b) for p in Py2]).transpose(shape_a)
    # Mask out terms which are duplicative
    mask = (
        np.diag(np.ones(Py.shape[-1], bool))
        + np.diag(np.ones(Py.shape[-1], bool), -1)[1:, 1:]
    )
    Py2 = Py2.transpose(shape_b)[mask.ravel()].transpose(shape_a)

    # This polynomial includes only cross terms
    P_obo = [
        np.expand_dims(p, axis=ndim) * Py.transpose(shape_b)[1:].transpose(shape_a)
        for p in Px.transpose(shape_b)[1:]
    ]
    P_obo = np.vstack([p.transpose(shape_b) for p in P_obo]).transpose(shape_a)
    # This matrix now has only the unique terms in the square of the polynomial
    P2 = np.vstack(
        [
            Px2.transpose(shape_b)[1:],
            Py2.transpose(shape_b)[1:],
            P_obo.transpose(shape_b),
        ]
    ).transpose(shape_a)
    return P, P2
