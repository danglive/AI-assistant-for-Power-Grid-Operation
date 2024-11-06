#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AgentRecoverTopo Module
=======================
This module implements the AgentRecoverTopo class, responsible for reverting substation topologies
back to their original configurations to reduce line overloads in the power grid.
It includes methods for finding the best line to reconnect, checking the legality of actions,
simulating actions, and determining valid actions based on the current grid observation.
"""

import logging
import re
from copy import deepcopy
from typing import Optional, List, Tuple, Iterator

import numpy as np
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.dtypes import dt_int
from grid2op.Agent import BaseAgent

# Setup logger for this module
logger = logging.getLogger(__name__)


class AgentRecoverTopo(BaseAgent):
    """
    AgentRecoverTopo is responsible for reverting substation topologies back to their original configurations
    to reduce line overloads in the power grid.
    """

    def __init__(self, env):
        super().__init__(env.action_space)
        self.env = env

    def find_best_line_to_reconnect(
        self, obs: BaseObservation, original_action: BaseAction
    ) -> BaseAction:
        """
        Finds the best line to reconnect to minimize the maximum line load (rho).
        """
        # Identify disconnected lines
        disconnected_lines = np.where(obs.line_status == False)[0]
        if len(disconnected_lines) == 0:
            return original_action

        min_rho = 10
        line_to_reconnect = -1

        # Iterate over each disconnected line
        for line in disconnected_lines:
            # Check if the line is not in cooldown
            if not obs.time_before_cooldown_line[line]:
                # Prepare an action to reconnect the line
                reconnect_array = np.zeros_like(obs.rho, dtype=int)
                reconnect_array[line] = 1
                reconnect_action = deepcopy(original_action)
                reconnect_action.update({"set_line_status": reconnect_array})

                # Verify if the action is legal
                if not self.is_legal(reconnect_action, obs):
                    continue

                # Simulate the action to assess its impact
                o, _, done, info = obs.simulate(reconnect_action)

                # Check if the action is valid
                if not self.is_valid(observation=obs, act=reconnect_action, done_sim=done, info_sim=info):
                    continue

                # Update if this action results in a lower maximum rho
                if o.rho.max() < min_rho:
                    line_to_reconnect = line
                    min_rho = o.rho.max()

        # Create the final action to reconnect the best line
        reconnect_out = deepcopy(original_action)
        if line_to_reconnect != -1:
            reconnect_array = np.zeros_like(obs.rho, dtype=int)
            reconnect_array[line_to_reconnect] = 1
            reconnect_out.update({"set_line_status": reconnect_array})

        return reconnect_out

    def is_legal(self, action: BaseAction, obs: BaseObservation) -> bool:
        """
        Checks if the action is legal by ensuring it doesn't involve elements in cooldown.
        """
        def extract_line_number(s: str) -> int:
            """
            Extracts the line number from a string key.
            """
            match = re.search(r'_(\d+)$', s)
            if match:
                return int(match.group(1))
            else:
                raise ValueError(f"Unexpected key format: {s}")

        action_dict = action.as_dict()

        # Empty actions are always legal
        if action_dict == {}:
            return True

        topo_action_type = list(action_dict.keys())[0]
        legal_act = True

        if topo_action_type in ["set_bus_vect", "change_bus_vect"]:
            substations = [int(sub) for sub in action_dict[topo_action_type]["modif_subs_id"]]

            for substation_to_operate in substations:
                # Check if the substation is in cooldown
                if obs.time_before_cooldown_sub[substation_to_operate]:
                    legal_act = False
                # Check associated lines for cooldown or disconnection
                for line in [
                    extract_line_number(key)
                    for key, val in action_dict[topo_action_type][str(substation_to_operate)].items()
                    if "line" in val["type"]
                ]:
                    if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                        legal_act = False

        elif topo_action_type == "set_line_status":
            # Check if any lines are in cooldown
            lines = [int(line) for line in action_dict[topo_action_type]["connected_id"]]
            for line in lines:
                if obs.time_before_cooldown_line[line]:
                    legal_act = False

        return legal_act

    def check_convergence(self, action: BaseAction, obs: BaseObservation) -> bool:
        """
        Checks if the grid simulation converges under stressed conditions after the action.
        """
        simulator = obs.get_simulator()
        # Increase load and generation by 5% to stress the grid
        load_p_stressed = obs.load_p * 1.05
        gen_p_stressed = obs.gen_p * 1.05

        # Predict the outcome under stressed conditions
        simulator_stressed = simulator.predict(
            act=action, new_gen_p=gen_p_stressed, new_load_p=load_p_stressed
        )

        return simulator_stressed.converged

    def get_from_dict_set_bus(self, original: dict) -> dict:
        """
        Converts a 'set_bus_vect' action dictionary into a 'set_bus' action dictionary.
        """
        dict_act = {"lines_or_id": [], "lines_ex_id": [], "loads_id": [], "generators_id": []}
        for key, value in original.items():
            for old, new in [
                ("line (origin)", "lines_or_id"),
                ("line (extremity)", "lines_ex_id"),
                ("load", "loads_id"),
                ("generator", "generators_id"),
            ]:
                if old == original[key]["type"]:
                    dict_act[new].append((int(key), int(value["new_bus"])))

        return {"set_bus": dict_act}

    def extract_action_set_from_actions(
        self, action_space: ActionSpace, action_vect: np.ndarray
    ) -> List[BaseAction]:
        """
        Extracts individual actions from a composite action vector.
        """
        action_set = []
        if not action_vect.any():
            return [action_space({})]

        act_dict = action_space.from_vect(action_vect).as_dict()
        if "set_bus_vect" in act_dict.keys():
            act_t = act_dict["set_bus_vect"]
            changed_sub_ids = act_t["modif_subs_id"]

            if len(changed_sub_ids) > 1:
                # Split the action into individual substation actions
                for sub_id in changed_sub_ids:
                    sub_action = action_space(self.get_from_dict_set_bus(act_t[sub_id]))
                    action_set.append(sub_action)
                return action_set
            else:
                return [action_space.from_vect(action_vect)]

        if "change_bus_vect" in act_dict.keys():
            act_t = act_dict["change_bus_vect"]
            if len(act_t["modif_subs_id"]) == 1:
                return [action_space.from_vect(action_vect)]
            else:
                raise NotImplementedError(
                    "Multiple substations were modified in the change_bus action. "
                    "This is not yet implemented in the tuple and triple approach. "
                    "Please use set_bus actions or unitary change_bus actions"
                )
        else:
            logging.warning(
                "An action was provided which could not be accounted for by the extract_action_set_from_actions method."
            )
            return [action_space.from_vect(action_vect)]

    def split_action_and_return(
        self, obs: BaseObservation, action_space: ActionSpace, action_vect: np.ndarray
    ) -> Iterator[BaseAction]:
        """
        Splits composite actions and yields them one by one, prioritizing those that minimize rho.
        """
        if not action_vect.any():
            yield action_space({})
            return

        split_actions = self.extract_action_set_from_actions(action_space, action_vect)
        for _ in range(len(split_actions)):
            obs_min = np.inf
            best_choice = None

            # Evaluate each split action to find the best one
            for act in split_actions:
                act_plus_reconnect = self.find_best_line_to_reconnect(obs, act)
                obs_f, _, done, info = obs.simulate(act_plus_reconnect)

                if not self.is_valid(observation=obs, act=act_plus_reconnect, done_sim=done, info_sim=info):
                    continue

                if obs_f.rho.max() < obs_min:
                    best_choice = act
                    obs_min = obs_f.rho.max()

            if best_choice is None:
                best_choice = action_space({})

            yield best_choice
            if best_choice in split_actions:
                split_actions.remove(best_choice)

    def is_valid(
        self,
        observation: BaseObservation,
        act: BaseAction,
        done_sim,
        info_sim,
        check_overload: Optional[bool] = False,
    ) -> bool:
        """
        Determines if an action is valid based on simulation results and legal constraints.
        """
        valid_action = True
        if done_sim:
            valid_action = False
        if not self.is_legal(act, observation):
            valid_action = False
        if info_sim.get("is_illegal", False):
            valid_action = False
        if info_sim.get("is_ambiguous", False):
            valid_action = False
        if any(info_sim.get("exception", [])):
            valid_action = False
        if check_overload and not self.check_convergence(act, observation):
            valid_action = False
        return valid_action

    def simulate_action(
        self, action_space: ActionSpace, obs: BaseObservation, action_vect: np.ndarray, check_overload: Optional[bool] = False
    ) -> Tuple[float, bool]:
        """
        Simulates an action and returns the maximum rho and whether the action is valid.
        """
        action = action_space.from_vect(action_vect)
        act_dict = action.as_dict()

        # Determine the appropriate simulation method based on the action type
        if "set_bus_vect" not in act_dict.keys():
            action = self.find_best_line_to_reconnect(obs=obs, original_action=action)
            obs_f, _, done, info = obs.simulate(action)
        elif len(act_dict["set_bus_vect"]["modif_subs_id"]) == 1:
            action = self.find_best_line_to_reconnect(obs=obs, original_action=action)
            obs_f, _, done, info = obs.simulate(action)
        else:
            gen = self.split_action_and_return(obs=obs, action_space=action_space, action_vect=action_vect)
            action = next(gen)
            obs_f, _, done, info = obs.simulate(action)

        rho_max = obs_f.rho.max()

        # Validate the action
        valid_action = self.is_valid(
            observation=obs, act=action, done_sim=done, info_sim=info, check_overload=check_overload
        )

        return rho_max, valid_action

    def revert_topo(
        self, action_space: ActionSpace, obs: BaseObservation, rho_limit: Optional[float] = 0.8
    ) -> np.ndarray:
        """
        Attempts to revert the topology to reduce line overloads by changing substations back to their original state.
        """
        act_best, obs_sim_rho, idx = None, None, None
        # Check if any substations have been modified and all lines are connected
        if np.any(obs.topo_vect == 2) and np.all(obs.line_status):
            available_actions = {}
            # Identify substations to revert
            for sub_id in range(obs.n_sub):
                if np.any(obs.sub_topology(sub_id) == 2):
                    action = action_space(
                        {
                            "set_bus": {
                                "substations_id": [
                                    (sub_id, np.ones(len(obs.sub_topology(sub_id)), dtype=dt_int))
                                ]
                            }
                        }
                    )
                    available_actions[sub_id] = action.to_vect()

            if any(available_actions):
                min_rho = min([obs.rho.max(), rho_limit])

                # Simulate actions to find the one that minimizes rho
                for sub_id, act_a in available_actions.items():
                    obs_sim_rho, valid_action = self.simulate_action(
                        action_vect=act_a, action_space=action_space, obs=obs
                    )
                    if not valid_action:
                        continue
                    if obs_sim_rho < min_rho:
                        idx = sub_id
                        act_best = act_a.copy()
                        min_rho = obs_sim_rho

        if act_best is not None:
            logging.info(
                f"Reverting Substation {idx} to original topology reduced rho from {obs.rho.max()} to {obs_sim_rho}"
            )
            return act_best
        else:
            # Return a do-nothing action if no improvement is found
            return action_space({}).to_vect()

    def act(self, observation: BaseObservation):
        """
        Determines the action to take based on the current observation by attempting to revert topology changes.
        """
        action_array = self.revert_topo(self.env.action_space, observation)
        return self.env.action_space.from_vect(action_array)