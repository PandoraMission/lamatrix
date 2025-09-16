"""Implements a special case of a lnGaussian2D that fits for SIP polynomials"""

from typing import Tuple

import numpy as np

from lamatrix import Constant, Polynomial, Sinusoid

from ..distributions import Distribution, DistributionsContainer
from ..math import MathMixins
from ..model import Model

try:
    from astropy.wcs import Sip as astropySIP
except ImportError as e:
    raise ImportError(
        "The 'sip' module requires astropy. Install it with:\n\n"
        "    pip install lamatrix[ffi]\n\n"
        "or if using Poetry:\n\n"
        "    poetry install --extras 'ffi'\n"
    ) from e


__all__ = ["SIP", "SIP1D", "AstrometryFitter"]


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
        posteriors=None,
        prior_A=(1, np.inf),
        prior_sigma_x=(1, np.inf),
        prior_sigma_y=(1, np.inf),
        prior_mu_x=None,
        prior_mu_y=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self.dx_name = dx_name
        self.dy_name = dy_name
        self.order = order
        self._validate_arg_names()

        super().__init__(
            priors=priors,
            posteriors=posteriors,
        )
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
            prior_sigma_x = self._validate_distributions(prior_sigma_x, width=1)
            prior_mu_x = self._validate_distributions(prior_mu_x, width=self.P_width)
            prior_sigma_y = self._validate_distributions(prior_sigma_y, width=1)
            prior_mu_y = self._validate_distributions(prior_mu_y, width=self.P_width)
            prior_A = self._validate_distributions(prior_A, width=1)
            self.priors = self.gaussian_parameters_to_coefficients(
                DistributionsContainer(
                    [
                        *[d.as_tuple() for d in prior_A],
                        *[d.as_tuple() for d in prior_sigma_x],
                        *[d.as_tuple() for d in prior_sigma_y],
                        *[d.as_tuple() for d in prior_mu_x],
                        *[d.as_tuple() for d in prior_mu_y],
                    ]
                )
            )
            self.priors = DistributionsContainer(
                [
                    *self.priors,
                    *DistributionsContainer.from_number(
                        self.width - (3 + 2 * self.P_width)
                    ),
                ]
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
    def P_width(self):
        return (self.order + 1) ** 2

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
        A = distributions[0]
        if isinstance(A, tuple):
            A, A_err = A
        sigma_x = distributions[1]
        if isinstance(sigma_x, tuple):
            sigma_x, sigma_x_err = sigma_x
        sigma_y = distributions[2]
        if isinstance(sigma_y, tuple):
            sigma_y, sigma_y_err = sigma_y

        mu_x = distributions[3 : 3 + self.P_width]
        if isinstance(mu_x[0], tuple):
            mu_x, mu_x_err = (
                DistributionsContainer(mu_x).mean,
                DistributionsContainer(mu_x).std,
            )
        else:
            mu_x = np.asarray(mu_x)
        mu_y = distributions[3 + self.P_width : 3 + self.P_width * 2]
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
            if sigma_x_err != np.inf:
                a_x_err = (1 / sigma_x**3) * sigma_x_err
            else:
                a_x_err = np.inf
            if (sigma_x_err != np.inf) & (np.all(mu_x_err != np.inf)):
                b_x_err = np.sqrt(
                    (1 / sigma_x**2 * mu_x_err) ** 2
                    + (-2 * mu_x / sigma_x**3 * sigma_x_err) ** 2
                )
            else:
                b_x_err = np.asarray([np.inf] * len(mu_x_err))
            if sigma_y_err != np.inf:
                a_y_err = (1 / sigma_y**3) * sigma_y_err
            else:
                a_y_err = np.inf
            if (sigma_y_err != np.inf) & (np.all(mu_y_err != np.inf)):
                b_y_err = np.sqrt(
                    (1 / sigma_y**2 * mu_y_err) ** 2
                    + (-2 * mu_y / sigma_y**3 * sigma_y_err) ** 2
                )
            else:
                b_y_err = np.asarray([np.inf] * len(mu_y_err))

            if (
                (sigma_x_err != np.inf)
                & (np.all(mu_x_err != np.inf))
                & (sigma_y_err != np.inf)
                & (np.all(mu_y_err != np.inf))
                & (A_err != np.inf)
            ):
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
            else:
                c_err = np.inf
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
        a_x = distributions[0]
        if isinstance(a_x, tuple):
            a_x, a_x_err = a_x
        a_y = distributions[1]
        if isinstance(a_y, tuple):
            a_y, a_y_err = a_y
        b_x = distributions[2 : 2 + self.P_width]
        if isinstance(b_x[0], tuple):
            b_x, b_x_err = (
                DistributionsContainer(b_x).mean,
                DistributionsContainer(b_x).std,
            )
        else:
            b_x = np.asarray(b_x)
        b_y = distributions[2 + self.P_width : 2 + self.P_width * 2]
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
            if a_x_err != np.inf:
                sigma_x_err = (1 / (4 * sigma_x * a_x**2)) * a_x_err
            else:
                sigma_x_err = np.inf
            if (a_x_err != np.inf) & (np.all(b_x_err != np.inf)):
                mu_x_err = np.sqrt(
                    (b_x / (2 * a_x**2) * a_x_err) ** 2
                    + (-1 / (2 * a_x) * b_x_err) ** 2
                )
            else:
                mu_x_err = np.asarray([np.inf] * len(b_x_err))

            if a_y_err != np.inf:
                sigma_y_err = (1 / (4 * sigma_y * a_y**2)) * a_y_err
            else:
                sigma_y_err = np.inf

            if (a_y_err != np.inf) & (np.all(b_y_err != np.inf)):
                mu_y_err = np.sqrt(
                    (b_y / (2 * a_y**2) * a_y_err) ** 2
                    + (-1 / (2 * a_y) * b_y_err) ** 2
                )
            else:
                mu_y_err = np.asarray([np.inf] * len(b_y_err))

            if (
                (a_x_err != np.inf)
                & (np.all(b_x_err != np.inf))
                & (a_y_err != np.inf)
                & (np.all(b_y_err != np.inf))
                & (c_err != np.inf)
            ):
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
            else:
                A_err = np.inf
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
        def process_str(var, idx):
            if idx == 0:
                return ""
            elif idx == 1:
                return f"{var}"
            else:
                return f"{var}^{idx}"

        P_str = [
            process_str(f"\\mathbf{{{self.latex_aliases[self.y_name]}}}", idx)
            + process_str(f"\\mathbf{{{self.latex_aliases[self.x_name]}}}", jdx)
            for idx in range(self.order + 1)
            for jdx in range(self.order + 1)
        ]
        P2_power = np.ravel(
            [i + np.arange(self.order + 1) for i in np.arange(self.order + 1)]
        )
        mask = (
            np.diag(np.ones(self.order + 1, bool))
            + np.diag(np.ones(self.order + 1, bool), -1)[1:, 1:]
        ).ravel()
        P2_power = P2_power[mask][1:]
        P2_str = [
            *[
                process_str(f"\\mathbf{{{self.latex_aliases[self.x_name]}}}", power)
                for power in P2_power
            ],
            *[
                process_str(f"\\mathbf{{{self.latex_aliases[self.y_name]}}}", power)
                for power in P2_power
            ],
            *[
                process_str(f"\\mathbf{{{self.latex_aliases[self.y_name]}}}", idx)
                + process_str(f"\\mathbf{{{self.latex_aliases[self.x_name]}}}", jdx)
                for idx in np.arange(1, self.order + 1)
                for jdx in np.arange(1, self.order + 1)
            ],
        ]

        return [
            f"\\mathbf{{{self.latex_aliases[self.dx_name]}}}^2",
            f"\\mathbf{{{self.latex_aliases[self.dy_name]}}}^2",
            *[
                f"\\mathbf{{{self.latex_aliases[self.dx_name]}}}"
                + (s if s != "" else "")
                for s in P_str
            ],
            *[
                f"\\mathbf{{{self.latex_aliases[self.dy_name]}}}"
                + (s if s != "" else "")
                for s in P_str
            ],
            *P2_str,
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
        return DistributionsContainer(
            self.coefficients_to_gaussian_parameters(self.posteriors)[
                3 : 3 + self.P_width
            ]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[
                3 : 3 + self.P_width
            ]
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
        return DistributionsContainer(
            self.coefficients_to_gaussian_parameters(self.posteriors)[
                3 + self.P_width : 3 + 2 * self.P_width
            ]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[
                3 + self.P_width : 3 + 2 * self.P_width
            ]
        )

    @property
    def A(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[0]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[0]
        )

    def mu_x_to_Model(self):
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

    def mu_y_to_Model(self):
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

    def to_astropySip(self, imshape: Tuple, crpix: Tuple):
        """
        Returns an astropy sip object with the best fit distortion coefficients.

        Parameters
        ----------
        imshape: Tuple
            The shape of the image to create the WCS for
        crpix: Tuple

        """
        if self.posteriors is None:
            raise ValueError("Posteriors are `None`.")
        iR, iC = np.mgrid[: imshape[0], : imshape[1]]
        fp_col, fp_row = iC - crpix[1], iR - crpix[0]
        pix_col, pix_row = (
            self.mu_y_to_Model().evaluate(**{self.x_name: iR, self.y_name: iC}) + iC,
            self.mu_x_to_Model().evaluate(**{self.x_name: iR, self.y_name: iC}) + iR,
        )

        A = np.asarray(
            [
                fp_row.ravel() ** idx * fp_col.ravel() ** jdx
                for idx in range(self.order + 1)
                for jdx in range(self.order + 1)
            ]
        ).T

        col_fp_to_pix = np.linalg.solve(
            A.T.dot(A), A.T.dot(pix_col.ravel() - fp_col.ravel() - crpix[1])
        ).reshape((self.order + 1, self.order + 1))
        row_fp_to_pix = np.linalg.solve(
            A.T.dot(A), A.T.dot(pix_row.ravel() - fp_row.ravel() - crpix[0])
        ).reshape((self.order + 1, self.order + 1))

        A = np.asarray(
            [
                (pix_row.ravel() - crpix[0]) ** idx
                * (pix_col.ravel() - crpix[1]) ** jdx
                for idx in range(self.order + 1)
                for jdx in range(self.order + 1)
            ]
        ).T

        col_pix_to_fp = np.linalg.solve(
            A.T.dot(A), A.T.dot(fp_col.ravel() - pix_col.ravel() + crpix[1])
        ).reshape((self.order + 1, self.order + 1))
        row_pix_to_fp = np.linalg.solve(
            A.T.dot(A), A.T.dot(fp_row.ravel() - pix_row.ravel() + crpix[0])
        ).reshape((self.order + 1, self.order + 1))

        csip = astropySIP(
            col_pix_to_fp, row_pix_to_fp, col_fp_to_pix, row_fp_to_pix, crpix
        )
        return csip


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


class SIP1D(SIP):
    """Special case of a SIP which is only one dimensional"""

    def __init__(
        self,
        t_name: str = "t",
        dx_name: str = "dx",
        dy_name: str = "dy",
        order: int = 1,
        priors=None,
        posteriors=None,
        prior_A=(1, np.inf),
        prior_sigma_x=(1, np.inf),
        prior_sigma_y=(1, np.inf),
        prior_mu_x=None,
        prior_mu_y=None,
    ):
        self.t_name = t_name
        self.dx_name = dx_name
        self.dy_name = dy_name
        self._validate_arg_names()

        super().__init__(
            priors=priors,
            posteriors=posteriors,
            order=order,
            prior_A=prior_A,
            prior_sigma_x=prior_sigma_x,
            prior_sigma_y=prior_sigma_y,
            prior_mu_x=prior_mu_x,
            prior_mu_y=prior_mu_y,
        )

    @property
    def width(self):
        return 3 + ((self.order + 1) * 2) + ((self.order) * 2)

    @property
    def P_width(self):
        return self.order + 1

    @property
    def nvectors(self):
        return 3

    @property
    def arg_names(self):
        return {self.t_name, self.dx_name, self.dy_name}

    @property
    def _initialization_attributes(self):
        return [
            "t_name",
            "dx_name",
            "dy_name",
            "order",
        ]

    @property
    def _equation(self):
        return ""

    def design_matrix(self, **kwargs):
        """Build a design matrix for SIP model.

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """

        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        t = kwargs.get(self.t_name)
        dx = kwargs.get(self.dx_name)
        dy = kwargs.get(self.dy_name)
        ndim = t.ndim
        shape_a = [*np.arange(1, ndim + 1).astype(int), 0]
        shape_b = [ndim, *np.arange(0, ndim)]

        p = Constant() + Polynomial("t", order=self.order)
        P = p.design_matrix(t=t)
        p2 = Polynomial("t", order=self.order * 2)
        P2 = p2.design_matrix(t=t)

        X = np.vstack(
            [
                np.expand_dims(dx, axis=ndim).transpose(shape_b) ** 2,
                np.expand_dims(dy, axis=ndim).transpose(shape_b) ** 2,
                (np.expand_dims(dx, axis=ndim) * P).transpose(shape_b),
                (np.expand_dims(dy, axis=ndim) * P).transpose(shape_b),
                P2.transpose(shape_b),
                np.ones((*t.shape, 1)).transpose(shape_b),
            ]
        ).transpose(shape_a)
        return X

    def mu_x_to_Model(self):
        """Convert the best fit mu_x values to an lamatrix.model.Polynomial object"""
        poly = Constant(posteriors=self.mu_x[0], priors=self.priors[3]) + Polynomial(
            self.t_name,
            order=self.order,
            priors=self.priors[3 + 1 : 3 + self.P_width],
            posteriors=self.mu_x[1:],
        )
        return poly

    def mu_y_to_Model(self):
        """Convert the best fit mu_x values to an lamatrix.model.Polynomial object"""
        poly = Constant(
            posteriors=self.mu_y[0], priors=self.priors[3 + self.P_width]
        ) + Polynomial(
            self.t_name,
            order=self.order,
            priors=self.priors[3 + self.P_width + 1 : 3 + 2 * self.P_width],
            posteriors=self.mu_y[1:],
        )
        return poly


class AstrometryFitter(SIP):
    """Special case of a SIP which fits astrometry..."""

    def __init__(
        self,
        t_name: str = "phi",
        dx_name: str = "dx",
        dy_name: str = "dy",
        order: int = 1,
        priors=None,
        posteriors=None,
        prior_A=(1, np.inf),
        prior_sigma_x=(1, np.inf),
        prior_sigma_y=(1, np.inf),
        prior_mu_x=None,
        prior_mu_y=None,
    ):
        self.t_name = t_name
        self.dx_name = dx_name
        self.dy_name = dy_name
        self._validate_arg_names()

        super().__init__(
            priors=priors,
            posteriors=posteriors,
            order=order,
            prior_A=prior_A,
            prior_sigma_x=prior_sigma_x,
            prior_sigma_y=prior_sigma_y,
            prior_mu_x=prior_mu_x,
            prior_mu_y=prior_mu_y,
        )

    @property
    def width(self):
        return 3 + ((self.order * 2) * 2) + ((self.order * 2) * 2)

    @property
    def P_width(self):
        return self.order * 2

    @property
    def nvectors(self):
        return 3

    @property
    def arg_names(self):
        return {self.t_name, self.dx_name, self.dy_name}

    @property
    def _initialization_attributes(self):
        return [
            "t_name",
            "dx_name",
            "dy_name",
            "order",
        ]

    @property
    def _equation(self):
        return ""

    def design_matrix(self, **kwargs):
        """Build a design matrix for SIP model.

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """

        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        t = kwargs.get(self.t_name)
        dx = kwargs.get(self.dx_name)
        dy = kwargs.get(self.dy_name)
        ndim = t.ndim
        shape_a = [*np.arange(1, ndim + 1).astype(int), 0]
        shape_b = [ndim, *np.arange(0, ndim)]

        p = Sinusoid("t", nterms=self.order)
        P = p.design_matrix(t=t)
        p2 = p**2
        P2 = p2.design_matrix(t=t)

        X = np.vstack(
            [
                np.expand_dims(dx, axis=ndim).transpose(shape_b) ** 2,
                np.expand_dims(dy, axis=ndim).transpose(shape_b) ** 2,
                (np.expand_dims(dx, axis=ndim) * P).transpose(shape_b),
                (np.expand_dims(dy, axis=ndim) * P).transpose(shape_b),
                P2.transpose(shape_b),
                np.ones((*t.shape, 1)).transpose(shape_b),
            ]
        ).transpose(shape_a)
        return X

    def mu_x_to_Model(self):
        """Convert the best fit mu_x values to an lamatrix.model.Polynomial object"""
        sinusoid = Sinusoid(
            self.t_name,
            nterms=self.order,
            priors=self.priors[2 : 2 + self.P_width],
            posteriors=self.mu_x,
        )
        return sinusoid

    def mu_y_to_Model(self):
        """Convert the best fit mu_x values to an lamatrix.model.Polynomial object"""
        sinusoid = Sinusoid(
            self.t_name,
            nterms=self.order,
            priors=self.priors[2 + self.P_width : 2 + 2 * self.P_width],
            posteriors=self.mu_y,
        )
        return sinusoid
