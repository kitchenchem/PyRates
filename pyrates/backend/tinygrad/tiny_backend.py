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

"""Wraps tinygrad such that it's low-level functions can be used by PyRates to create and simulate a compute graph.
"""

# pyrates internal _imports
from ..base import BaseBackend
from ..computegraph import ComputeVar
from .tiny_funcs import tiny_funcs

# external _imports
from tinygrad import Tensor, dtypes
import numpy as np
from typing import Callable, Optional, Dict, List

# meta infos
__status__ = "development"


#######################################
# classes for backend functionalities #
#######################################

class TinyBackend(BaseBackend):

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates tinygrad backend.
        """

        # add user-provided operations to function dict
        tiny_ops = tiny_funcs.copy()
        if ops:
            tiny_ops.update(ops)

        # call parent method
        super().__init__(ops=tiny_ops, imports=imports, **kwargs)
        self._imports[0] = "from math import pi, sqrt"

    def get_var(self, v: ComputeVar):
        return Tensor(super().get_var(v).tolist())

    @staticmethod
    def _solve_euler(func: Callable, args: tuple, T: float, dt: float, dts: float, y: Tensor, t0: Tensor
                     ) -> np.ndarray:

        # preparations for fixed step-size integration
        t0 = int(t0.item()) if isinstance(t0, Tensor) else int(t0)
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        state_rec = np.zeros((store_steps, y.shape[0]) if y.shape else (store_steps, 1), dtype=np.float32)

        # solve ivp for forward Euler method
        for step in range(t0, steps + t0):
            if step % store_step == t0:
                state_rec[idx, :] = y.numpy()
                idx += 1
            rhs = func(step, y, *args)
            y = y + dt * rhs

        return state_rec

    @staticmethod
    def _solve_heun(func: Callable, args: tuple, T: float, dt: float, dts: float, y: Tensor, t0: Tensor
                    ) -> np.ndarray:

        # preparations for fixed step-size integration
        t0 = int(t0.item()) if isinstance(t0, Tensor) else int(t0)
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        state_rec = np.zeros((store_steps, y.shape[0]) if y.shape else (store_steps, 1), dtype=np.float32)

        # solve ivp for Heun's method
        for step in range(t0, steps + t0):
            if step % store_step == t0:
                state_rec[idx, :] = y.numpy()
                idx += 1
            rhs = func(step, y, *args)
            y_0 = y + dt * rhs
            y = y + dt / 2 * (rhs + func(step, y_0, *args))

        return state_rec

    @staticmethod
    def _solve_scipy(func: Callable, args: tuple, T: float, dt: float, y: Tensor, t0: Tensor,
                     times: np.ndarray, **kwargs) -> np.ndarray:

        # solve ivp via scipy methods
        from scipy.integrate import solve_ivp
        t0 = float(t0.item()) if isinstance(t0, Tensor) else float(t0)
        kwargs['t_eval'] = times

        # wrapper to rhs function
        def f(t, y_np):
            rhs = func(Tensor([t]), Tensor(y_np.tolist()), *args)
            return rhs.numpy()

        # call scipy solver
        results = solve_ivp(fun=f, t_span=(t0, T), y0=y.numpy(), first_step=dt, **kwargs)
        return results['y'].T
