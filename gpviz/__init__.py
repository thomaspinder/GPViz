from .gp import plot
from .kernel import plot
from .utils import tidy_legend, glow
import matplotlib as mpl
import pkg_resources

__version__ = "0.0.2"
__author__ = 'Thomas Pinder'

# Add GPViz stylesheet to matplotlib's stylesheet set.
data_path = pkg_resources.resource_filename('gpviz', 'styles/')
gpviz_stylesheets = mpl.style.core.read_style_directory(data_path)
mpl.style.core.update_nested_dict(mpl.style.library, gpviz_stylesheets)
