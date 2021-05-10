import matplotlib.pyplot as plt
from gpviz import install_stylesheet


def test_install():
    install_stylesheet()
    styles = plt.style.available
    assert 'gpviz' in styles
