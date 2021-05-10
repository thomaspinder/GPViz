from gpviz.styles.colours import get_colours, get_cmap
import matplotlib as mpl


def test_get_colours():
    cs = get_colours()
    assert len(list(cs.keys())) > 0


def test_get_cmap():
    cm = get_cmap()
    assert isinstance(cm, mpl.colors.LinearSegmentedColormap)