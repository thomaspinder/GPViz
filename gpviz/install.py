from pkg_resources import resource_filename
import matplotlib as mpl
import os
import shutil


def install_stylesheet():
    stylelib = os.path.join(mpl.get_configdir(), 'stylelib')

    # in case you don't have a stylelib directory yet
    try:
        os.mkdir(stylelib)

    except FileExistsError:
        pass

    try:
        filepath = resource_filename('gpviz', 'styles/gpviz.mplstyle')
    except ModuleNotFoundError:
        filepath = resource_filename(__name__, 'styles/gpviz.mplstyle')

    shutil.copy(filepath, stylelib)
