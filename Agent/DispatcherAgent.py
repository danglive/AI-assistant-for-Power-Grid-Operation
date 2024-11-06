#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DispatcherAgent Module
======================
This module implements the DispatcherAgent class, which is responsible for managing
the dispatching actions in the grid. It computes the optimal redispatching, curtailment,
and storage actions to maintain the grid within safe operational limits.

The DispatcherAgent uses convex optimization (via CVXPY) to solve the dispatch problem,
taking into account the current grid state and constraints.
"""

import copy
import logging
import warnings

import cvxpy as cp
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction
from grid2op.Backend import PandaPowerBackend
from grid2op.l2rpn_utils.idf_2023 import ObservationIDF2023
from lightsim2grid import LightSimBackend
from lightsim2grid.gridmodel import init

# Setup logger for this module
logger = logging.getLogger(__name__)


class DispatcherAgent(BaseAgent):
    """
    DispatcherAgent class computes the optimal dispatch actions (redispatching, curtailment,
    storage) to maintain grid stability and prevent overloads.

    Args:
        env (grid2op.Environment.BaseEnv): The Grid2Op environment.
        action_space (grid2op.Action.ActionSpace): The action space of the environment.
        config (dict): Configuration settings.
        time_step (int, optional): Simulation time step. Defaults to 1.
        verbose (int, optional): Verbosity level for logging. Defaults to 1.
    """

    SOLVER_TYPES = [cp.OSQP, cp.SCS, cp.SCIPY]

    def __init__(self, env, action_space, config, time_step=1, verbose=1):
        super().__init__(action_space)
        self.env = env
        self.do_nothing = action_space({})
        self.config = config
        self.time_step = time_step
        self.verbose = verbose
        self._get_grid_info(env)
        self._init_params(env)
        self.max_iter = config["max_iter"]
        self.flow_computed = np.full(env.n_line, np.NaN, dtype=float)

    def _get_grid_info(self, env):
        """Retrieve grid information from the environment."""
        self.n_line = env.n_line
        self.n_sub = env.n_sub
        self.n_load = env.n_load
        self.n_gen = env.n_gen
        self.n_storage = env.n_storage
        self.line_or_to_subid = copy.deepcopy(env.line_or_to_subid)
        self.line_ex_to_subid = copy.deepcopy(env.line_ex_to_subid)
        self.load_to_subid = copy.deepcopy(env.load_to_subid)
        self.gen_to_subid = copy.deepcopy(env.gen_to_subid)
        self.storage_to_subid = copy.deepcopy(env.storage_to_subid)
        self.storage_Emax = copy.deepcopy(env.storage_Emax)

    def _init_params(self, env):
        """Initialize optimization parameters and variables."""
        self.margin_rounding = float(self.config["margin_rounding"])
        self.margin_sparse = float(self.config["margin_sparse"])
        self.rho_danger = float(self.config["rho_danger"])
        self.rho_safe = float(self.config["rho_safe"])

        # Initialize CVXPY parameters
        self._margin_th_limit = cp.Parameter(
            value=self.config["margin_th_limit"], nonneg=True
        )
        self._penalty_curtailment_unsafe = cp.Parameter(
            value=self.config["penalty_curtailment_unsafe"], nonneg=True
        )
        self._penalty_redispatching_unsafe = cp.Parameter(
            value=self.config["penalty_redispatching_unsafe"], nonneg=True
        )
        self._penalty_storage_unsafe = cp.Parameter(
            value=self.config["penalty_storage_unsafe"], nonneg=True
        )
        self._penalty_curtailment_safe = cp.Parameter(
            value=self.config["penalty_curtailment_safe"], nonneg=True
        )
        self._penalty_redispatching_safe = cp.Parameter(
            value=self.config["penalty_redispatching_safe"], nonneg=True
        )
        self._penalty_storage_safe = cp.Parameter(
            value=self.config["penalty_storage_safe"], nonneg=True
        )
        self._weight_redisp_target = cp.Parameter(
            value=self.config["weight_redisp_target"], nonneg=True
        )
        self._weight_storage_target = cp.Parameter(
            value=self.config["weight_storage_target"], nonneg=True
        )
        self._weight_curtail_target = cp.Parameter(
            value=self.config["weight_curtail_target"], nonneg=True
        )
        self._alpha_por_error = cp.Parameter(
            value=self.config["alpha_por_error"], nonneg=True
        )

        self.nb_max_bus = 2 * self.n_sub
        self._storage_setpoint = 0.5 * self.storage_Emax
        SoC = np.zeros(shape=self.nb_max_bus)
        for bus_id in range(self.nb_max_bus):
            SoC[bus_id] = 0.5 * self._storage_setpoint[self.storage_to_subid == bus_id].sum()
        self._storage_target_bus = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * SoC, nonneg=True
        )
        self._storage_power_obs = cp.Parameter(value=0.0)

        # Get powerline impedance parameters
        powerlines_x, powerlines_g, powerlines_b, powerlines_ratio = self._get_powerline_impedance(env)
        self._powerlines_x = cp.Parameter(shape=powerlines_x.shape, value=1.0 * powerlines_x, pos=True)
        self._powerlines_g = cp.Parameter(shape=powerlines_g.shape, value=1.0 * powerlines_g, pos=True)
        self._powerlines_b = cp.Parameter(shape=powerlines_b.shape, value=1.0 * powerlines_b, neg=True)
        self._powerlines_ratio = cp.Parameter(shape=powerlines_ratio.shape, value=1.0 * powerlines_ratio, pos=True)
        self._prev_por_error = cp.Parameter(shape=powerlines_x.shape, value=np.zeros(env.n_line))

        # Voltage magnitude parameters
        self.vm_or = cp.Parameter(shape=self.n_line, value=np.ones(self.n_line), pos=True)
        self.vm_ex = cp.Parameter(shape=self.n_line, value=np.ones(self.n_line), pos=True)

        # Bus parameters
        self.bus_or = cp.Parameter(shape=self.n_line, value=1 * self.line_or_to_subid, integer=True)
        self.bus_ex = cp.Parameter(shape=self.n_line, value=1 * self.line_ex_to_subid, integer=True)
        self.bus_load = cp.Parameter(shape=self.n_load, value=1 * self.load_to_subid, integer=True)
        self.bus_gen = cp.Parameter(shape=self.n_gen, value=1 * self.gen_to_subid, integer=True)
        self.bus_storage = cp.Parameter(shape=self.n_storage, value=1 * self.storage_to_subid, integer=True)

        # Injection parameters per bus
        zeros_bus = np.zeros(self.nb_max_bus)
        self.load_per_bus = cp.Parameter(shape=self.nb_max_bus, value=1.0 * zeros_bus, nonneg=True)
        self.gen_per_bus = cp.Parameter(shape=self.nb_max_bus, value=1.0 * zeros_bus, nonneg=True)

        # Constraints per bus
        self.redisp_up = cp.Parameter(shape=self.nb_max_bus, value=1.0 * zeros_bus, nonneg=True)
        self.redisp_down = cp.Parameter(shape=self.nb_max_bus, value=1.0 * zeros_bus, nonneg=True)
        self.curtail_down = cp.Parameter(shape=self.nb_max_bus, value=1.0 * zeros_bus, nonneg=True)
        self.curtail_up = cp.Parameter(shape=self.nb_max_bus, value=1.0 * zeros_bus, nonneg=True)
        self.storage_down = cp.Parameter(shape=self.nb_max_bus, value=1.0 * zeros_bus, nonneg=True)
        self.storage_up = cp.Parameter(shape=self.nb_max_bus, value=1.0 * zeros_bus, nonneg=True)

        # Thermal limits
        self._th_lim_mw = cp.Parameter(shape=self.n_line, value=1.0 * env.get_thermal_limit(), nonneg=True)

        # Past dispatch and state of charge
        self._past_dispatch = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus))
        self._past_state_of_charge = cp.Parameter(shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)

    def _get_powerline_impedance(self, env):
        """Retrieve powerline impedance parameters from the environment's backend."""
        if isinstance(env.backend, LightSimBackend):
            line_info = env.backend._grid.get_lines()
            trafo_info = env.backend._grid.get_trafos()
        elif isinstance(env.backend, PandaPowerBackend):
            pp_net = env.backend._grid
            grid_model = init(pp_net)
            line_info = grid_model.get_lines()
            trafo_info = grid_model.get_trafos()
        else:
            raise RuntimeError(
                f"Unknown backend type: {type(env.backend)}. "
                "If you want to use OptimCVXPY, you need to provide the reactance "
                "of each powerline/transformer in per unit in the lines_x parameter."
            )

        powerlines_x = np.array(
            [float(el.x_pu) for el in line_info] + [float(el.x_pu) for el in trafo_info]
        )
        powerlines_g = np.array(
            [(1 / (el.r_pu + 1j * el.x_pu)).real for el in line_info]
            + [(1 / (el.r_pu + 1j * el.x_pu)).real for el in trafo_info]
        )
        powerlines_b = np.array(
            [(1 / (el.r_pu + 1j * el.x_pu)).imag for el in line_info]
            + [(1 / (el.r_pu + 1j * el.x_pu)).imag for el in trafo_info]
        )
        powerlines_ratio = np.array([1.0] * len(line_info) + [el.ratio for el in trafo_info])
        return powerlines_x, powerlines_g, powerlines_b, powerlines_ratio

    def _update_topo_param(self, obs: ObservationIDF2023):
        """Update topology parameters based on the current observation."""
        tmp_ = 1 * obs.line_or_to_subid
        tmp_[obs.line_or_bus == 2] += obs.n_sub
        self.bus_or.value[:] = tmp_

        tmp_ = 1 * obs.line_ex_to_subid
        tmp_[obs.line_ex_bus == 2] += obs.n_sub
        self.bus_ex.value[:] = tmp_

        self.bus_ex.value[(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0
        self.bus_or.value[(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0

        tmp_ = 1 * obs.load_to_subid
        tmp_[obs.load_bus == 2] += obs.n_sub
        self.bus_load.value[:] = tmp_

        tmp_ = 1 * obs.gen_to_subid
        tmp_[obs.gen_bus == 2] += obs.n_sub
        self.bus_gen.value[:] = tmp_

        if self.bus_storage is not None:
            tmp_ = 1 * obs.storage_to_subid
            tmp_[obs.storage_bus == 2] += obs.n_sub
            self.bus_storage.value[:] = tmp_

        # Update voltage magnitudes
        self.vm_or.value[:] = np.array(
            [v_or / 138 if v_or < 147 else v_or / 161 if v_or < 171 else v_or / 345 for v_or in obs.v_or]
        )
        self.vm_ex.value[:] = np.array(
            [v_ex / 138 if v_ex < 147 else v_ex / 161 if v_ex < 171 else v_ex / 345 for v_ex in obs.v_ex]
        )

    def _update_th_lim_param(self, obs: ObservationIDF2023):
        """Update thermal limit parameters based on the current observation."""
        threshold_ = 1.0
        self._th_lim_mw.value[:] = (
            (0.001 * obs.thermal_limit) ** 2 * obs.v_or ** 2 * 3.0 - obs.q_or ** 2
        )
        mask_ok = self._th_lim_mw.value >= threshold_
        self._th_lim_mw.value[mask_ok] = np.sqrt(self._th_lim_mw.value[mask_ok])
        self._th_lim_mw.value[~mask_ok] = threshold_

    def _update_storage_power_obs(self, obs: ObservationIDF2023):
        """Update storage power observation."""
        self._storage_power_obs.value = 0.0

    def _update_inj_param(self, obs: ObservationIDF2023):
        """Update injection parameters per bus based on the current observation."""
        self.load_per_bus.value[:] = 0.0
        self.gen_per_bus.value[:] = 0.0
        load_p = 1.0 * obs.load_p
        load_p *= (obs.gen_p.sum() - self._storage_power_obs.value) / load_p.sum()
        for bus_id in range(self.nb_max_bus):
            self.load_per_bus.value[bus_id] += load_p[self.bus_load.value == bus_id].sum()
            self.gen_per_bus.value[bus_id] += obs.gen_p[self.bus_gen.value == bus_id].sum()

    def _add_redisp_const_per_bus(self, obs: ObservationIDF2023, bus_id: int):
        """Add redispatching constraints per bus."""
        self.redisp_up.value[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
        self.redisp_down.value[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()

    def _add_storage_const_per_bus(self, obs: ObservationIDF2023, bus_id: int):
        """Add storage constraints per bus."""
        if self.bus_storage is None:
            return
        if obs.storage_max_p_prod is not None:
            stor_down = obs.storage_max_p_prod[self.bus_storage.value == bus_id].sum()
            stor_down = min(
                stor_down,
                obs.storage_charge[self.bus_storage.value == bus_id].sum() * (60.0 / obs.delta_time),
            )
            self.storage_down.value[bus_id] = stor_down
        else:
            self.storage_down.value[bus_id] = 0.0

        if obs.storage_max_p_absorb is not None:
            stor_up = obs.storage_max_p_absorb[self.bus_storage.value == bus_id].sum()
            stor_up = min(
                stor_up,
                (
                    obs.storage_Emax - obs.storage_charge
                )[self.bus_storage.value == bus_id].sum()
                * (60.0 / obs.delta_time),
            )
            self.storage_up.value[bus_id] = stor_up
        else:
            self.storage_up.value[bus_id] = 0.0

    def _remove_margin_rounding(self):
        """Adjust constraints to account for margin rounding."""
        self.storage_down.value[self.storage_down.value > self.margin_rounding] -= self.margin_rounding
        self.storage_up.value[self.storage_up.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_down.value[self.curtail_down.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_up.value[self.curtail_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_up.value[self.redisp_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_down.value[self.redisp_down.value > self.margin_rounding] -= self.margin_rounding

    def _update_constraints_param_unsafe(self, obs: ObservationIDF2023):
        """Update constraints parameters for unsafe conditions."""
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.0
        for bus_id in range(self.nb_max_bus):
            self._add_redisp_const_per_bus(obs, bus_id)
            mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
            self.curtail_down.value[bus_id] = 0.0
            self.curtail_up.value[bus_id] = tmp_[mask_].sum()
            self._add_storage_const_per_bus(obs, bus_id)
        self._remove_margin_rounding()

    def _validate_param_values(self):
        """Validate parameter values to ensure they are within acceptable bounds."""
        self.storage_down._validate_value(self.storage_down.value)
        self.storage_up._validate_value(self.storage_up.value)
        self.curtail_down._validate_value(self.curtail_down.value)
        self.curtail_up._validate_value(self.curtail_up.value)
        self.redisp_up._validate_value(self.redisp_up.value)
        self.redisp_down._validate_value(self.redisp_down.value)
        self._th_lim_mw._validate_value(self._th_lim_mw.value)
        self._storage_target_bus._validate_value(self._storage_target_bus.value)
        self._past_dispatch._validate_value(self._past_dispatch.value)
        self._past_state_of_charge._validate_value(self._past_state_of_charge.value)

    def update_parameters(self, obs: ObservationIDF2023, safe: bool = False):
        """Update all parameters based on the current observation."""
        self._update_topo_param(obs)
        self._update_th_lim_param(obs)
        self._update_inj_param(obs)
        if safe:
            # Function `_update_constraints_param_safe` is not defined in the provided code
            # Assuming it's similar to `_update_constraints_param_unsafe`
            self._update_constraints_param_unsafe(obs)
        else:
            self._update_constraints_param_unsafe(obs)
        self._validate_param_values()

    def _aux_compute_kcl(self, inj_bus, f_or):
        """Compute Kirchhoff's Current Law (KCL) equations for each bus."""
        KCL_eq = []
        for bus_id in range(self.nb_max_bus):
            tmp = inj_bus[bus_id]
            if np.any(self.bus_or.value == bus_id):
                tmp += cp.sum(f_or[self.bus_or.value == bus_id])
            if np.any(self.bus_ex.value == bus_id):
                tmp -= cp.sum(f_or[self.bus_ex.value == bus_id])
            KCL_eq.append(tmp)
        return KCL_eq

    def _mask_theta_zero(self):
        """Identify buses where the voltage angle (theta) can be set to zero."""
        theta_is_zero = np.full(self.nb_max_bus, True, bool)
        theta_is_zero[self.bus_or.value] = False
        theta_is_zero[self.bus_ex.value] = False
        theta_is_zero[self.bus_load.value] = False
        theta_is_zero[self.bus_gen.value] = False
        if self.bus_storage is not None:
            theta_is_zero[self.bus_storage.value] = False
        theta_is_zero[0] = True  # Reference bus
        return theta_is_zero

    def _solve_problem(self, prob, solver_type=None):
        """Attempt to solve the optimization problem using specified solvers."""
        if solver_type is None:
            for solver in DispatcherAgent.SOLVER_TYPES:
                res = self._solve_problem(prob, solver_type=solver)
                if res:
                    if self.verbose:
                        logger.info(f"Solver {solver} has converged. Stopping solver search now.")
                    return True
            return False
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if solver_type is cp.OSQP:
                    prob.solve(solver=solver_type, verbose=0, warm_start=False, max_iter=self.max_iter)
                elif solver_type is cp.SCS:
                    prob.solve(solver=solver_type, warm_start=False, max_iters=1000)
                else:
                    prob.solve(solver=solver_type, warm_start=False)
            if np.isfinite(prob.value):
                return True
            else:
                logger.warning(f"Problem diverged with solver {solver_type}, infinite value returned")
                raise cp.error.SolverError("Infinite value")
        except cp.error.SolverError as exc_:
            logger.warning(f"Problem diverged with solver {solver_type}: {exc_}")
            return False

    def run_dc(self, obs: ObservationIDF2023):
        """Run a DC power flow calculation to estimate line flows."""
        self._update_topo_param(obs)
        self._update_inj_param(obs)
        theta = cp.Variable(shape=self.nb_max_bus)
        f_or = cp.multiply(1.0 / self._powerlines_x, (theta[self.bus_or.value] - theta[self.bus_ex.value]))
        inj_bus = self.load_per_bus - self.gen_per_bus
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()
        constraints = ([theta[theta_is_zero] == 0] + [el == 0 for el in KCL_eq])
        cost = 1.0
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob, solver_type=cp.OSQP)
        if has_converged:
            self.flow_computed[:] = f_or.value
        else:
            logger.error(
                "Problem diverged with DC approximation for all solvers "
                f"({DispatcherAgent.SOLVER_TYPES}). Is your grid connected (one single connected component)?"
            )
            self.flow_computed[:] = np.NaN
        return has_converged

    def reset(self, obs: ObservationIDF2023):
        """Reset the agent's state at the beginning of a new episode."""
        self._prev_por_error.value[:] = 0.0
        converged = self.run_dc(obs)
        if converged:
            self._prev_por_error.value[:] = self.flow_computed - obs.p_or
        else:
            logger.warning(
                "Impossible to initialize the DispatcherAgent because the DC powerflow did not converge."
            )

    def compute_optimum_unsafe(self):
        """Compute the optimal dispatch actions under unsafe conditions."""
        theta = cp.Variable(shape=self.nb_max_bus)
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)
        storage = cp.Variable(shape=self.nb_max_bus)
        redispatching = cp.Variable(shape=self.nb_max_bus)

        f_or = cp.multiply(
            1.0 / self._powerlines_x,
            (theta[self.bus_or.value] - theta[self.bus_ex.value]),
        )
        f_or_corr = f_or - self._alpha_por_error * self._prev_por_error
        inj_bus = (
            self.load_per_bus + storage
            - (self.gen_per_bus + redispatching - curtailment_mw)
        )
        energy_added = (
            cp.sum(curtailment_mw)
            + cp.sum(storage)
            - cp.sum(redispatching)
            - self._storage_power_obs
        )
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()

        constraints = (
            [theta[theta_is_zero] == 0]
            + [el == 0 for el in KCL_eq]
            + [redispatching <= self.redisp_up, redispatching >= -self.redisp_down]
            + [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down]
            + [storage <= self.storage_up, storage >= -self.storage_down]
            + [energy_added == 0]
        )

        cost = (
            self._penalty_curtailment_unsafe * cp.sum_squares(curtailment_mw)
            + self._penalty_storage_unsafe * cp.sum_squares(storage)
            + self._penalty_redispatching_unsafe * cp.sum_squares(redispatching)
            + cp.sum_squares(
                cp.pos(cp.abs(f_or_corr) - self._margin_th_limit * self._th_lim_mw)
            )
        )

        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob, solver_type=cp.OSQP)
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
            self._storage_power_obs.value = 0.0
        else:
            logger.error(
                "compute_optimum_unsafe: Problem diverged. No continuous action will be applied."
            )
            self.flow_computed[:] = np.NaN
            zeros_bus = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * zeros_bus, 1.0 * zeros_bus, 1.0 * zeros_bus)
        return res

    def _clean_vect(self, curtailment, storage, redispatching):
        """Clean vectors by setting values below the sparse margin to zero."""
        curtailment[np.abs(curtailment) < self.margin_sparse] = 0.0
        storage[np.abs(storage) < self.margin_sparse] = 0.0
        redispatching[np.abs(redispatching) < self.margin_sparse] = 0.0

    def to_grid2op(
        self,
        obs: ObservationIDF2023,
        curtailment: np.ndarray,
        storage: np.ndarray,
        redispatching: np.ndarray,
        base_action: BaseAction = None,
        safe=False,
    ) -> BaseAction:
        """Convert optimization results into a Grid2Op action."""
        self._clean_vect(curtailment, storage, redispatching)
        if base_action is None:
            base_action = self.action_space()
        if base_action.n_storage and np.any(np.abs(storage) > 0.0):
            storage_ = np.zeros(shape=base_action.n_storage)
            storage_[:] = storage[self.bus_storage.value]
            base_action.storage_p = storage_
        if np.any(np.abs(curtailment) > 0.0):
            curtailment_mw = np.zeros(shape=base_action.n_gen) - 1.0
            gen_curt = obs.gen_renewable & (obs.gen_p > 0.1)
            idx_gen = self.bus_gen.value[gen_curt]
            tmp_ = curtailment[idx_gen]
            modif_gen_optim = tmp_ != 0.0
            aux_ = curtailment_mw[gen_curt]
            aux_[modif_gen_optim] = (
                obs.gen_p[gen_curt][modif_gen_optim]
                - tmp_[modif_gen_optim]
                * obs.gen_p[gen_curt][modif_gen_optim]
                / self.gen_per_bus.value[idx_gen][modif_gen_optim]
            )
            aux_[~modif_gen_optim] = -1.0
            curtailment_mw[gen_curt] = aux_
            curtailment_mw[~gen_curt] = -1.0
            if safe:
                gen_id_max = (curtailment_mw >= obs.gen_p_before_curtail) & obs.gen_renewable
                if np.any(gen_id_max):
                    curtailment_mw[gen_id_max] = base_action.gen_pmax[gen_id_max]
            base_action.curtail_mw = curtailment_mw
        elif safe and np.abs(self.curtail_down.value).max() == 0.0:
            vect = 1.0 * base_action.gen_pmax
            vect[~obs.gen_renewable] = -1.0
            base_action.curtail_mw = vect
        if np.any(np.abs(redispatching) > 0.0):
            redisp_ = np.zeros(obs.n_gen)
            gen_redi = obs.gen_redispatchable
            idx_gen = self.bus_gen.value[gen_redi]
            tmp_ = redispatching[idx_gen]
            redisp_avail = np.zeros(self.nb_max_bus)
            for bus_id in range(self.nb_max_bus):
                if redispatching[bus_id] > 0.0:
                    redisp_avail[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
                elif redispatching[bus_id] < 0.0:
                    redisp_avail[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
            prop_to_gen = np.zeros(obs.n_gen)
            redisp_up = np.zeros(obs.n_gen, dtype=bool)
            redisp_up[gen_redi] = tmp_ > 0.0
            prop_to_gen[redisp_up] = obs.gen_margin_up[redisp_up]
            redisp_down = np.zeros(obs.n_gen, dtype=bool)
            redisp_down[gen_redi] = tmp_ < 0.0
            prop_to_gen[redisp_down] = obs.gen_margin_down[redisp_down]
            nothing_happens = (redisp_avail[idx_gen] == 0.0) & (prop_to_gen[gen_redi] == 0.0)
            set_to_one_nothing = 1.0 * redisp_avail[idx_gen]
            set_to_one_nothing[nothing_happens] = 1.0
            redisp_avail[idx_gen] = set_to_one_nothing
            if np.any(np.abs(redisp_avail[idx_gen]) <= self.margin_sparse):
                logger.warning(
                    "Some generators have a dispatch assigned by the optimizer, "
                    "but they don't have any margin. The dispatch has been canceled "
                    "(this was probably caused by the optimizer not meeting certain constraints)."
                )
                this_fix_ = 1.0 * redisp_avail[idx_gen]
                too_small_here = np.abs(this_fix_) <= self.margin_sparse
                tmp_[too_small_here] = 0.0
                this_fix_[too_small_here] = 1.0
                redisp_avail[idx_gen] = this_fix_
            redisp_[gen_redi] = tmp_ * prop_to_gen[gen_redi] / redisp_avail[idx_gen]
            redisp_[~gen_redi] = 0.0
            base_action.redispatch = redisp_
        return base_action

    def act(self, observation, act, reward=None, done: bool = False):
        """Decide on the action to take based on the current observation."""
        curtailment, storage, redispatching = self.compute_optimum_unsafe()
        act = self.to_grid2op(observation, curtailment, storage, redispatching, base_action=act, safe=False)
        return act