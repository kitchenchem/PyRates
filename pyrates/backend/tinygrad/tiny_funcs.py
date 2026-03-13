# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network model_templates and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation

"""Contains tinygrad function definitions that may be used for PyRates model equations.
"""

# external _imports
import numpy as np

# meta infos
__status__ = "development"

# function definitions
######################

sigmoid = lambda x: 1./(1. + np.exp(-x))

interp = """
def interp(x_new, x, y):
    idx = (x - x_new).abs().argmin()
    cond = x[idx].abs() > x_new.abs()
    i1 = cond.where(idx - 1, idx)
    i2 = cond.where(idx, idx + 1)
    return (y[i1] + y[i1]) * 0.5
"""

# dictionary for backend import
###############################

tinygrad_ = 'pyrates.backend.tinygrad.tiny_wrappers'

tiny_funcs = {
    'maxi': {'call': 'maximum', 'func': np.maximum, 'imports': [f'{tinygrad_}.maximum']},
    'mini': {'call': 'minimum', 'func': np.minimum, 'imports': [f'{tinygrad_}.minimum']},
    'round': {'call': 'round', 'func': np.round, 'imports': [f'{tinygrad_}.round']},
    'sum': {'call': 'sum', 'func': np.sum, 'imports': [f'{tinygrad_}.sum']},
    'mean': {'call': 'mean', 'func': np.mean, 'imports': [f'{tinygrad_}.mean']},
    'matmul': {'call': 'matmul', 'func': np.dot, 'imports': [f'{tinygrad_}.matmul']},
    'matvec': {'call': 'matmul', 'func': np.dot, 'imports': [f'{tinygrad_}.matmul']},
    'roll': {'call': 'roll', 'func': np.roll, 'imports': [f'{tinygrad_}.roll']},
    'randn': {'call': 'randn', 'func': np.random.randn, 'imports': [f'{tinygrad_}.randn']},
    'tanh': {'call': 'tanh', 'func': np.tanh, 'imports': [f'{tinygrad_}.tanh']},
    'sinh': {'call': 'sinh', 'func': np.sinh, 'imports': [f'{tinygrad_}.sinh']},
    'cosh': {'call': 'cosh', 'func': np.cosh, 'imports': [f'{tinygrad_}.cosh']},
    'arctan': {'call': 'arctan', 'func': np.arctan, 'imports': [f'{tinygrad_}.arctan']},
    'arcsin': {'call': 'arcsin', 'func': np.arcsin, 'imports': [f'{tinygrad_}.arcsin']},
    'arccos': {'call': 'arccos', 'func': np.arccos, 'imports': [f'{tinygrad_}.arccos']},
    'sin': {'call': 'sin', 'func': np.sin, 'imports': [f'{tinygrad_}.sin']},
    'cos': {'call': 'cos', 'func': np.cos, 'imports': [f'{tinygrad_}.cos']},
    'tan': {'call': 'tan', 'func': np.tan, 'imports': [f'{tinygrad_}.tan']},
    'exp': {'call': 'exp', 'func': np.exp, 'imports': [f'{tinygrad_}.exp']},
    'sigmoid': {'call': 'sigmoid', 'func': sigmoid, 'imports': [f'{tinygrad_}.sigmoid']},
    'interp': {'call': 'interp', 'func': np.interp, 'def': interp, 'imports': []},
    'absv': {'call': 'abs', 'func': np.abs, 'imports': [f'{tinygrad_}.abs']},
    'sign': {'call': 'sign', 'func': np.sign, 'imports': [f'{tinygrad_}.sign']},
    'log': {'call': 'log', 'func': np.log, 'imports': [f'{tinygrad_}.log']},
    'concatenate': {'call': 'concat', 'func': np.concatenate, 'imports': [f'{tinygrad_}.concat']},
}
