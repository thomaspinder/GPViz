from setuptools import setup, find_packages


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


# Optional Packages
EXTRAS = {
    "dev": ["black", "isort", "pylint", "flake8",],
    "tests": ["pytest",],
    "docs": [
        "furo==2020.12.30b24",
        "nbsphinx==0.8.1",
        "nb-black==1.0.7",
        "matplotlib==3.3.3",
        "sphinx-copybutton==0.3.5",
    ],
}

setup(
    name="GPViz",
    version="0.0.4",
    author="Thomas Pinder",
    author_email="t.pinder2@lancaster.ac.uk",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="A Python package for Gaussian process visualisation.",
    long_description="GPViz provides a convenient interface to GPJax that facilitates flexible plotting of Gaussian process priors, posterios, kernels and datasets.",
    install_requires=parse_requirements_file("requirements.txt"),
    extras_require=EXTRAS,
    keywords=["gaussian-processes jax machine-learning bayesian"],
)
