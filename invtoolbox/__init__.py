"""Inverse Problem Toolbox: linear and nonlinear inverse problems, variational / iterative /
Bayesian reconstruction methods, parameter-choice rules, and a PyQt5 GUI."""
from .core.methods import METHODS, Result, run
from .core.paramchoice import sweep
from .core.problems import REGISTRY, Data, Problem, build, categories

__version__ = "1.0.0"
__all__ = ["METHODS", "REGISTRY", "Data", "Problem", "Result", "build", "categories", "run", "sweep"]
