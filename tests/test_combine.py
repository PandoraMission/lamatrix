import numpy as np
import pytest
from lamatrix import Polynomial, Constant, Sinusoid
from lamatrix.combine import JointModel, CrosstermModel


def test_bad():
    """Can not add two constants together"""
    g1 = Constant()
    g2 = Constant()
    with pytest.raises(ValueError):
        g = g1 + g2


def test_add():
    g1 = Constant()
    g2 = Polynomial("x", polyorder=3)
    g = g1 + g2
    assert isinstance(g, JointModel)
    assert g.width == 4
    assert len(g.models) == 2
    g = g1 + g2 + g2
    assert isinstance(g, JointModel)
    assert g.width == 7
    assert len(g.models) == 3
    g = g1 + g2 + g1
    g = (g1 + g2) + (g1 + g2)
    assert isinstance(g, JointModel)
    assert g.width == 7
    assert len(g.models) == 3


def test_priors():
    g1 = Constant()
    g2 = Polynomial("x", polyorder=3)
    g = g1 + g2

    g.set_prior(0, (0, 10))
    g.set_priors([(0, 10), (0, np.inf), (0, np.inf), (0, np.inf)])

    g1 = Constant()
    g2 = Polynomial("x", polyorder=3)
    g = g1 + g2

    assert g[0].prior_distributions == [(0, np.inf)]
    g[0].set_prior(0, (0, 10))
    assert g[0].prior_distributions == [(0, 10)]
    assert g.prior_distributions[0] == (0, 10)

    assert g[1].prior_distributions == [(0, np.inf), (0, np.inf), (0, np.inf)]
    g[1].set_prior(1, (0, 10))
    assert g[1].prior_distributions[1] == (0, 10)
    assert g.prior_distributions[0] == (0, 10)
    assert g.prior_distributions[2] == (0, 10)


def test_polynomial():
    g1 = Constant()
    g2 = Polynomial("x", polyorder=3)
    g = g1 + g2
    assert hasattr(g, "arg_names")
    assert g.arg_names == {"x"}
    dm = g.design_matrix(x=np.arange(10))
    assert dm.shape == (10, 4)
    assert (dm[:, 0] == np.ones(10)).all()
    assert g.width == 4
    assert len(g.prior_distributions) == 4
    assert len(g.fit_distributions) == 4
    assert isinstance(g.prior_distributions, list)
    assert isinstance(g.prior_distributions[0], tuple)

    x = np.arange(-1, 1, 0.01)
    w, c = np.random.normal(size=2)
    y = w * x + c
    y += np.random.normal(0, 0.001, size=x.shape[0])
    ye = np.ones_like(x) + 0.001

    g.fit(data=y, errors=ye, x=x)
    assert len(g.fit_distributions) == 4
    assert np.isclose(g.fit_distributions[1][0], w, atol=0.01)


def test_cross():
    g1 = Polynomial("x", polyorder=3)
    g2 = Polynomial("y", polyorder=4)
    g3 = Polynomial("z", polyorder=2)

    g = CrosstermModel(g1, g2)
    assert len(g.prior_distributions) == 12
    assert g.width == 12
    assert g.prior_mean.shape == (12,)
    assert g.prior_std.shape == (12,)
    assert g.arg_names == {"x", "y"}

    x, y = np.mgrid[0:20, 0:21] - 10
    dm = g.design_matrix(x=x.ravel(), y=y.ravel())
    assert dm.shape == (x.ravel().shape[0], g.width)

    g = CrosstermModel(g1, g2, g3)
    assert len(g.prior_distributions) == 24
    assert g.width == 24
    assert g.prior_mean.shape == (24,)
    assert g.prior_std.shape == (24,)
    assert g.arg_names == {"x", "y", "z"}

    x, y, z = np.mgrid[0:20, 0:21, :22] - 10
    dm = g.design_matrix(x=x.ravel(), y=y.ravel(), z=z.ravel())
    assert dm.shape == (x.ravel().shape[0], g.width)


def test_math():
    """Tests all the math combinations"""
    c = Constant()
    g1 = Polynomial("x", polyorder=3)
    g2 = Polynomial("y", polyorder=4)
    g3 = Polynomial("z", polyorder=2)
    g4 = Polynomial("a", polyorder=1)

    # Constants can't be combined
    # c + c --> ValueError
    with pytest.raises(ValueError):
        c + c

    # Constants can be added to models
    # g + c --> SIG(g, c)
    # c + g --> SIG(c, g)

    assert isinstance(g1 + c, JointModel)
    assert isinstance(c + g1, JointModel)
    assert (g1 + c)[1].arg_names == {}
    assert (c + g1)[0].arg_names == {}

    # Models multiplied by constants return themselves
    # g * c --> g
    # c * g --> g

    assert isinstance(g1 * c, Polynomial)
    assert isinstance(c * g1, Polynomial)
    assert g1 * c == g1
    assert c * g1 == g1

    # Models add to a SIG
    # g1 + g2 --> SIG(g1, g2)
    # g2 + g1 --> SIG(g2, g1)

    assert isinstance(g1 + g2, JointModel)
    assert isinstance(g2 + g1, JointModel)
    assert (g1 + g2)[0].arg_names == {"x"}
    assert (g2 + g1)[0].arg_names == {"y"}

    # Models multiple to a CG
    # g1 * g2 --> CG(g1, g2)
    # g2 * g1 --> CG(g2, g1)

    assert isinstance(g1 * g2, CrosstermModel)
    assert isinstance(g2 * g1, CrosstermModel)
    assert (g1 * g2).models[0].arg_names == {"x"}
    assert (g2 * g1).models[0].arg_names == {"y"}

    # SIGs with constants return themselves when another constant is added
    # SIG(g1, c) + c = SIG(g1, c)
    # c + SIG(g1, c) = SIG(g1, c)

    assert isinstance((g1 + c) + c, JointModel)
    assert isinstance((c + g1) + c, JointModel)
    assert ((c + g1) + c).width == 4
    assert (c + g1 + c).width == 4

    # SIG multiplied by constants return themselves
    # SIG(g1, g2) * c = SIG(g1, g2)
    # c * SIG(g1, g2) = SIG(g1, g2)
    # SIG(g1, c) * c = SIG(g1, c)
    # c * SIG(g1, c) = SIG(g1, c)

    assert isinstance((g1 + c) * c, JointModel)
    assert isinstance((c + g1) * c, JointModel)
    assert ((c + g1) * c).width == 4
    assert (c + g1 * c).width == 4
    assert isinstance((g1 + g2) * c, JointModel)
    assert isinstance((g2 + g1) * c, JointModel)
    assert ((g2 + g1) * c).width == 7
    assert (g2 + g1 * c).width == 7

    # SIGs plus a generator appends the generator, preserving the order
    # SIG(g1, g2) + g3 = SIG(g1, g2, g3)
    # g3 + SIG(g1, g2) = SIG(g3, g1, g2)
    assert isinstance(g1 + g2 + g3, JointModel)
    assert len((g1 + g2 + g3).models) == 3
    assert len(((g1 + g2) + g3).models) == 3
    assert len((g3 + (g1 + g2)).models) == 3

    # SIGs multiplied by a generator creates a SIG of CGs
    # SIG(g1, g2) * g3 = SIG(CG(g1, g3), CG(g1, g3))
    # g3 * SIG(g1, g2) = SIG(CG(g1, g3), CG(g1, g3))

    assert isinstance(((g1 + g2) * g3), JointModel)
    assert isinstance((g3 * (g1 + g2)), JointModel)
    assert isinstance(((g1 + g2) * g3)[0], CrosstermModel)
    assert isinstance((g3 * (g1 + g2))[1], CrosstermModel)
    assert isinstance(((g1 + g2) * g3)[1], CrosstermModel)
    assert isinstance((g3 * (g1 + g2))[0], CrosstermModel)

    # CG plus a generator creates a SIG
    # CG(g1, g2) + g3 = SIG(CG(g1, g2), g3)
    # g3 + CG(g1, g2) = SIG(g3, CG(g1, g2))

    assert isinstance((g1 * g2) + g3, JointModel)
    assert isinstance(g3 + (g1 * g2), JointModel)
    assert isinstance(((g1 * g2) + g3)[0], CrosstermModel)
    assert isinstance((g1 + (g2 * g3))[1], CrosstermModel)

    # CG multiplied by a generator creates a CCG
    # CG(g1, g2) * g3 = CG(g1, g2, g3)
    # g3 * CG(g1, g2) = CG(g3, g1, g2)

    assert isinstance((g1 * g2) * g3, CrosstermModel)
    assert isinstance(g3 * (g1 * g2), CrosstermModel)
    assert ((g1 * g2) * g3).models[0].arg_names == {"x"}
    assert (g1 * (g2 * g3)).models[0].arg_names == {"x"}
    assert (g2 * (g1 * g3)).models[0].arg_names == {"y"}

    # CGs multiplied with constants return themselves
    assert isinstance((g1 * g2) * c, CrosstermModel)
    assert ((g1 * g2) * c).width == 12

    # SIG plus a SIG create a SIG
    # SIG(g1, g2) + SIG(g3, g4) = SIG(g1, g2, g3, g4)
    # SIG(g3, g4) + SIG(g1, g2) = SIG(g3, g4, g1, g2)

    g = (g1 + g2) + (g3 + g4)
    assert isinstance(g, JointModel)
    assert len(g.models) == 4
    assert g[0].arg_names == {"x"}
    g = (g3 + g4) + (g1 + g2)
    assert g[0].arg_names == {"z"}

    # SIG plus a CG create a SIG
    # SIG(g1, g2) + CG(g3, g4) = SIG(g1, g2, CG(g3, g4))
    # CG(g1, g2) + SIG(g3, g4) = SIG(CG(g1, g2), g3, g4)

    g = (g1 + g2) + (g3 * g4)
    assert isinstance(g, JointModel)
    assert len(g.models) == 3
    assert g[0].arg_names == {"x"}
    assert g[2].arg_names == {"z", "a"}
    g = (g1 * g2) + (g3 + g4)
    assert isinstance(g, JointModel)
    assert len(g.models) == 3
    assert g[0].arg_names == {"x", "y"}
    assert g[2].arg_names == {"a"}

    # SIG multiplied by a SIG create a SIG of crossterms
    # SIG(g1, g2) * SIG(g3, g4) = SIG(CC(g1, g3), CC(g1, g4), CC(g2, g3), CC(g2, g4))
    g = (g1 + g2) * (g3 + g4)
    assert isinstance(g, JointModel)
    assert len(g.models) == 4
    assert np.all([isinstance(gen, CrosstermModel) for gen in g.models])
    assert g[0].models[0].arg_names == {"x"} and g[0].models[1].arg_names == {"z"}
    assert g[1].models[0].arg_names == {"x"} and g[1].models[1].arg_names == {"a"}
    assert g[2].models[0].arg_names == {"y"} and g[2].models[1].arg_names == {"z"}
    assert g[3].models[0].arg_names == {"y"} and g[3].models[1].arg_names == {"a"}
