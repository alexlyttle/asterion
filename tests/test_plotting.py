import pytest, warnings
import asterion as ast
from asterion.data import get_example_results

DATA = get_example_results()

@pytest.mark.parametrize(
    "kwargs", 
    [
        {},
        {"group": "prior"},
        {"kind": "background"},
        {"kind": "glitchless"},
        {"delta_nu": 110.01}
    ]
)
def test_echelle(kwargs):
    """Test plotting an echelle diagram."""
    ax = ast.plot_echelle(DATA, **kwargs)
    group = kwargs.get("group", "posterior")
    if "delta_nu" in kwargs:
        dnu = kwargs["delta_nu"]
    else:
        if group == "prior":
            dnu = DATA["prior_predictive"]["delta_nu"].median().to_numpy()
        else:
            dnu = DATA[group]["delta_nu"].median().to_numpy()

    # Test axis labels
    assert (ax.xaxis.label.get_text()
            == "$\\nu\\,\\mathrm{mod}.\\,{" + f"{dnu:.2f}" + "}$/$\\mathrm{\\mu Hz}$")
    assert (ax.yaxis.label.get_text()
            == "$\\nu$/$\\mathrm{\\mu Hz}$")
    
    # Test observed data pltot
    _, labels = ax.get_legend_handles_labels()
    if group == "posterior":
        assert r"$\nu_\mathrm{obs}$" in labels

@pytest.mark.parametrize(
    "kwargs", 
    [
        {},
        {"group": "prior"},
        {"kind": "He"},
        {"kind": "CZ"},
    ]
)
def test_glitch(kwargs):
    """Test plotting the glitch."""
    ax = ast.plot_glitch(DATA, **kwargs)
    group = kwargs.get("group", "posterior")

    # Test axis labels
    assert (ax.xaxis.label.get_text() == "$n$")
    assert ax.yaxis.label.get_text().startswith("$\\delta\\nu$")
    # assert ax.yaxis.label.get_text().endswith("\\mathrm{\\mu Hz}$")
    
    # Test observed data in plot
    _, labels = ax.get_legend_handles_labels()
    if group == "posterior":
        r"$\nu_\mathrm{obs}$" in labels

# def test_corner(kwargs):
    