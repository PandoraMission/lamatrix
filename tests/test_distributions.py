import numpy as np

from lamatrix import DistributionsContainer


def test_distribution_container():
    distributions = [(0, np.inf), (1, 0.3), (1, 0.2)]
    priors = DistributionsContainer(distributions)
    assert len(priors) == 3
    assert priors[1] == (1, 0.3)
    priors[1] = (2, 0.5)
    assert priors[1] == (2, 0.5)
    priors[:] = [(1, 10), (1, 10), (1, 10)]
    assert priors[0] == (1, 10)
    assert priors[1] == (1, 10)
    assert priors[2] == (1, 10)
