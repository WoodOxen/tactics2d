import numpy as np


class Spiral:
    """This class implements a spiral interpolation."""

    def __init__(self, gamma: float):
        """Initialize the spiral interpolation.

        Args:
            gamma (float): The curvature of the spiral.
        """
        self.gamma = gamma

    def get_curve(self, start_point, kappa: float = 0, theta: float = 0) -> np.ndarray:
        """This method returns the interpolated points of the spiral.

        [TODO]: Correct the implementation of the spiral interpolation.

        Args:
            start_point (_type_): _description_
            kappa (float, optional): _description_. Defaults to 0.
            theta (float, optional): _description_. Defaults to 0.

        Returns:
            The interpolated points of the spiral. The shape is (n_interpolation, 2).
        """
        x_start, y_start = start_point

        # start
        C0 = x_start + 1j * y_start

        if self.gamma == 0 and kappa == 0:
            # straight line
            Cs = C0 + np.exp(1j * theta * s)

        elif self.gamma == 0 and kappa != 0:
            # circular arc
            Cs = (
                C0
                + (1 / kappa) * np.exp(1j * theta) / kappa * np.sin(kappa * s)
                + 1j * (1 - np.cos(kappa * s))
            )

        else:
            # fresnel integrals
            Sa, Ca = fresnel((kappa + self.gamma * s) / np.sqrt(np.pi * np.abs(self.gamma)))
            Sb, Cb = fresnel((kappa + self.gamma * s) / np.sqrt(np.pi * np.abs(self.gamma)))
            Cs1 = np.sqrt(np.pi * np.abs(self.gamma)) * np.exp(
                1j * (theta - kappa**2 / 2 / self.gamma)
            )
            Cs2 = np.sign(self.gamma) * (Ca - Cb) + 1j * (Sa - Sb)

            Cs = C0 + Cs1 * Cs2

        theta = self._gamma * s**2 / 2 + kappa * s + theta

        return
