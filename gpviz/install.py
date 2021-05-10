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

    # copy the files into the stylelib directory
    try:
        shutil.copy('gpviz/styles/gpviz.mplstyle', stylelib)
    except FileNotFoundError:
        shutil.copy('styles/gpviz.mplstyle', stylelib)
