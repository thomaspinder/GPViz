import matplotlib.colors as mcolors


def get_colours():
    colours = {
        "base": "#B5121B",
        "black": "#2F2F2F",
        "dark_gray": "#254441",
        "light_gray": "#D9D7DD",
        "cool_gray": "#9197AE",
        "eggshell": "#FAF3DD",
        "off_white": "#FFF8F0",
        "blue": "#23B5D3",
        "orange": "#F77F00",
        "green": "#0B6E4F",
        "indigo": "#083D77",
    }
    return colours


def get_cmap():
    levs = range(256)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        name="red_white_blue",
        colors=[(0.980, 0.952, 0.866), (0.709, 0.070, 0.105)],
        N=len(levs) - 1,
    )
    return cmap
