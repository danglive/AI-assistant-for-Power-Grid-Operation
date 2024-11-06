#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AgentTopology Module
====================
This module implements the AgentTopology class, responsible for selecting the best topology-changing actions
to mitigate overloads in the power grid. It includes methods for splitting actions by zones, recovering reference
topology, ranking overloads, and choosing appropriate actions based on the current grid state.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Set
from grid2op.Agent import BaseAgent
from grid2op.l2rpn_utils.idf_2023 import ObservationIDF2023
from grid2op.Action import BaseAction, ActionSpace

logger = logging.getLogger(__name__)

class ZoneStrategy(ABC):
    """Base class for different zone handling strategies"""
    @abstractmethod
    def get_zone_actions(
        self,
        zones: Set[int],
        topo_act_list_by_area_n1: Dict[str, List[BaseAction]],
        topo_act_list_by_area_overload: Dict[str, List[BaseAction]]
    ) -> Tuple[List[BaseAction], List[BaseAction]]:
        """Get actions for zones based on strategy"""
        pass

class SingleAgentStrategy(ZoneStrategy):
    """Treats entire grid as single zone"""
    def get_zone_actions(
        self,
        zones: Set[int],
        topo_act_list_by_area_n1: Dict[str, List[BaseAction]],
        topo_act_list_by_area_overload: Dict[str, List[BaseAction]]
    ) -> Tuple[List[BaseAction], List[BaseAction]]:
        # Combine all actions regardless of zones
        action_overload = []
        action_n1 = []
        for zone_actions in topo_act_list_by_area_overload.values():
            action_overload.extend(zone_actions)
        for zone_actions in topo_act_list_by_area_n1.values():
            action_n1.extend(zone_actions)
        return action_overload, action_n1

class MultiAgentIndependentStrategy(ZoneStrategy):
    """Each zone acts independently"""
    def get_zone_actions(
        self,
        zones: Set[int],
        topo_act_list_by_area_n1: Dict[str, List[BaseAction]],
        topo_act_list_by_area_overload: Dict[str, List[BaseAction]]
    ) -> Tuple[List[BaseAction], List[BaseAction]]:
        action_overload = []
        action_n1 = []
        # Only get actions from overloaded zones
        for zone in zones:
            zone_key = f"zone_{zone}"
            action_overload.extend(topo_act_list_by_area_overload.get(zone_key, []))
            action_n1.extend(topo_act_list_by_area_n1.get(zone_key, []))
        return action_overload, action_n1

class MultiAgentDependentStrategy(ZoneStrategy):
    """Zones coordinate actions based on priority"""
    def get_zone_actions(
        self,
        zones: Set[int],
        topo_act_list_by_area_n1: Dict[str, List[BaseAction]],
        topo_act_list_by_area_overload: Dict[str, List[BaseAction]]
    ) -> Tuple[List[BaseAction], List[BaseAction]]:
        action_overload = []
        action_n1 = []
        # Process overloaded zones first, then others
        all_zones = range(len(topo_act_list_by_area_n1))
        zone_order = list(zones) + [z for z in all_zones if z not in zones]
        
        for zone in zone_order:
            zone_key = f"zone_{zone}"
            action_overload.extend(topo_act_list_by_area_overload.get(zone_key, []))
            action_n1.extend(topo_act_list_by_area_n1.get(zone_key, []))
        return action_overload, action_n1

class AgentTopology(BaseAgent):
    """
    AgentTopology class is responsible for determining the optimal topology actions to alleviate grid overloads.

    Args:
        env (grid2op.Environment.BaseEnv): The Grid2Op environment.
        action_space (grid2op.Action.ActionSpace): The action space of the environment.
        config (dict): Configuration settings.
        time_step (int, optional): Simulation time step. Defaults to 1.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Attributes:
        env (grid2op.Environment.BaseEnv): The Grid2Op environment.
        do_nothing (grid2op.Action.BaseAction): Action that does nothing.
        config (dict): Configuration settings.
        time_step (int): Simulation time step.
        verbose (int): Verbosity level.
        line_to_sub_id (dict): Mapping between zones and line IDs.
        areas_by_sub_id (dict): Mapping of substations by area.
    """
    
    def __init__(
        self,
        env,
        action_space: ActionSpace,
        config: dict,
        time_step: int = 1,
        verbose: int = 1,
    ):
        super().__init__(action_space)
        self.env = env
        self.do_nothing = action_space({})
        self.config = config
        self.time_step = time_step
        self.verbose = verbose
        
        # Initialize zone-related data
        self.line_to_sub_id = env._game_rules.legal_action.lines_id_by_area
        self.areas_by_sub_id = {}
        self._initialize_areas_by_sub_id()
        
        # Initialize zone strategies
        self.strategies = {
            "single_agent": SingleAgentStrategy(),
            "multi_agent_independent": MultiAgentIndependentStrategy(),
            "multi_agent_dependent": MultiAgentDependentStrategy()
        }
        
    def _initialize_areas_by_sub_id(self):
        """Initializes mapping of substations by area"""
        try:
            self.areas_by_sub_id = self.env._game_rules.legal_action.substations_id_by_area
            logger.debug("Initialized areas_by_sub_id successfully")
        except AttributeError as e:
            logger.error(f"Failed to initialize areas_by_sub_id: {e}")
            self.areas_by_sub_id = {}

    def _split_actions_by_zone(
        self,
        action_space_n1: List[BaseAction],
        action_space_overload: List[BaseAction],
    ) -> Tuple[Dict[str, List[BaseAction]], Dict[str, List[BaseAction]]]:
        """
        Splits actions based on the zones (areas) of substations.

        Args:
            action_space_n1 (List[BaseAction]): The N-1 action space (for non-overload situations).
            action_space_overload (List[BaseAction]): The overload action space (for overload situations).

        Returns:
            Tuple[Dict[str, List[BaseAction]], Dict[str, List[BaseAction]]]:
                - topo_act_list_by_area_n1: Dictionary of N-1 actions by zone.
                - topo_act_list_by_area_overload: Dictionary of overload actions by zone.
        """
        num_zones = len(self.areas_by_sub_id)
        topo_act_list_by_area_overload = {f"zone_{i}": [] for i in range(num_zones)}
        topo_act_list_by_area_n1 = {f"zone_{i}": [] for i in range(num_zones)}

        # Classify overload actions by zone
        for act in action_space_overload:
            sub_id = self._get_action_substation_id(act)
            if sub_id is not None:
                for zone, sub_ids in self.areas_by_sub_id.items():
                    if sub_id in sub_ids:
                        topo_act_list_by_area_overload[f"zone_{zone}"].append(act)
                        break

        # Classify N-1 actions by zone
        for act in action_space_n1:
            sub_id = self._get_action_substation_id(act)
            if sub_id is not None:
                for zone, sub_ids in self.areas_by_sub_id.items():
                    if sub_id in sub_ids:
                        topo_act_list_by_area_n1[f"zone_{zone}"].append(act)
                        break

        return topo_act_list_by_area_n1, topo_act_list_by_area_overload

    def _get_action_substation_id(self, action: BaseAction) -> Optional[int]:
        """
        Extracts the substation ID from an action.

        Args:
            action (BaseAction): The action to extract the substation ID from.

        Returns:
            Optional[int]: The substation ID if available, otherwise None.
        """
        try:
            if hasattr(action, 'as_dict'):
                sub_id = int(action.as_dict()["set_bus_vect"]["modif_subs_id"][0])
                return sub_id
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to get substation ID from action: {e}")
        return None

    def _get_ranked_overloads(self, observation: ObservationIDF2023) -> List[int]:
        """
        Ranks the list of overloads by rho value, from highest to lowest.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.

        Returns:
            List[int]: List of line IDs that are overloaded.
        """
        overloaded_indices = np.where(observation.rho >= self.config.get("rho_danger", 0.99))[0]
        sorted_overloads = overloaded_indices[np.argsort(-observation.rho[overloaded_indices])]
        return sorted_overloads.tolist()

    def _check_zone(self, line_id: int) -> Optional[int]:
        """Determines zone for given line"""
        for zone, ids in self.line_to_sub_id.items():
            if line_id in ids:
                return zone
        return None

    def _get_overload_zones(self, observation: ObservationIDF2023) -> Set[int]:
        """
        Identifies the zones experiencing overload based on the current observation.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.

        Returns:
            Set[int]: A set of zones that are currently overloaded.
        """
        overload_line_ids = self._get_ranked_overloads(observation)
        zones = {self._check_zone(line_id) for line_id in overload_line_ids}
        zones.discard(None)
        return zones

    def _get_actions_from_zones(
        self,
        observation: ObservationIDF2023,
        action_space_n1: List[BaseAction],
        action_space_overload: List[BaseAction]
    ) -> Tuple[List[BaseAction], List[BaseAction]]:
        """
        Retrieves actions from zones based on overload or N-1 situations.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            action_space_n1 (List[BaseAction]): The N-1 action space.
            action_space_overload (List[BaseAction]): The overload action space.

        Returns:
            Tuple[List[BaseAction], List[BaseAction]]:
                - action_overload: List of overload actions.
                - action_n1: List of N-1 actions.
        """
        topo_act_list_by_area_n1, topo_act_list_by_area_overload = self._split_actions_by_zone(
            action_space_n1, action_space_overload
        )
        
        zones = self._get_overload_zones(observation)
        search_method = self.config.get("search_method", "single_agent")
        
        # Get strategy and execute
        strategy = self.strategies.get(search_method)
        if strategy is None:
            logger.error(f"Invalid search method: {search_method}")
            return [], []
            
        return strategy.get_zone_actions(
            zones,
            topo_act_list_by_area_n1,
            topo_act_list_by_area_overload
        )

    def recover_reference_topology(
        self,
        observation: ObservationIDF2023,
        base_action: BaseAction,
        min_rho: Optional[float],
    ) -> Tuple[Optional[BaseAction], Optional[ObservationIDF2023], float]:
        """
        Attempts to recover the initial topology state based on the environment's parameters.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            base_action (BaseAction): Base action from the agent.
            min_rho (Optional[float]): Minimum rho value initially (to decide actions).

        Returns:
            Tuple[Optional[BaseAction], Optional[ObservationIDF2023], float]:
                - action_chosen: The selected action (if any).
                - obs_chosen: The observation corresponding to the chosen action.
                - min_rho: Updated minimum rho value after choosing an action.
        """
        if min_rho is None:
            min_rho = float(observation.rho.max())

        action_chosen = None
        obs_chosen = None

        ref_actions = self.env.action_space.get_back_to_ref_state(observation).get('substation', None)
        if ref_actions:
            for action in ref_actions:
                sub_id = self._get_action_substation_id(action)
                if sub_id is not None and observation.time_before_cooldown_sub[sub_id] == 0:
                    combined_action = base_action + action
                    obs_simu, _, _, info = observation.simulate(combined_action, time_step=self.time_step)
                    if not info.get('exception', False) and obs_simu.rho.max() < min_rho:
                        action_chosen = action
                        obs_chosen = obs_simu
                        min_rho = float(obs_simu.rho.max())
                        break

        return action_chosen, obs_chosen, min_rho

    def change_substation_topology(
        self,
        observation: ObservationIDF2023,
        base_action: BaseAction,
        action_space_n1: List[BaseAction],
        action_space_overload: List[BaseAction],
        min_rho: float,
    ) -> Tuple[
        Optional[BaseAction],
        Optional[ObservationIDF2023],
        float,
        List[float],
        List[BaseAction],
        List[int],
        List[ObservationIDF2023],
    ]:
        """
        Attempts to change the substation topology to alleviate overloads.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            base_action (BaseAction): Base action from the agent.
            action_space_n1 (List[BaseAction]): N-1 action space.
            action_space_overload (List[BaseAction]): Overload action space.
            min_rho (float): Minimum rho value to beat.

        Returns:
            Tuple containing:
                - action_chosen (Optional[BaseAction]): The selected action.
                - obs_chosen (Optional[ObservationIDF2023]): The observation corresponding to the chosen action.
                - min_rho (float): Updated minimum rho value.
                - rho_list (List[float]): List of rho values from simulations.
                - list_action (List[BaseAction]): List of actions tried.
                - list_nb_modif_objects (List[int]): Number of modified objects for each action.
                - obs_list (List[ObservationIDF2023]): List of simulated observations.
        """
        action_overload, action_n1 = self._get_actions_from_zones(
            observation, action_space_n1, action_space_overload
        )

        alertable_lines = observation.alertable_line_ids
        disconnected_lines = np.where(observation.rho == 0)[0]
        has_alertable = np.any(np.isin(disconnected_lines, alertable_lines))

        topo_actions = action_n1 + action_overload if has_alertable else action_overload + action_n1

        rho_list = []
        list_action = []
        list_nb_modif_objects = []
        obs_list = []
        action_chosen = None
        obs_chosen = None

        for action in topo_actions:
            substation_id = self._get_action_substation_id(action)
            if substation_id is None or observation.time_before_cooldown_sub[substation_id] > 0:
                continue

            combined_action = base_action + action
            try:
                obs_simu, _, _, info = observation.simulate(combined_action, time_step=self.time_step)
                if not info.get('exception', False) and obs_simu.rho.max() <= min_rho:
                    rho_list.append(float(obs_simu.rho.max()))
                    list_action.append(action)
                    nb_modif_objects = action.as_dict()["set_bus_vect"]["nb_modif_objects"]
                    list_nb_modif_objects.append(nb_modif_objects)
                    obs_list.append(obs_simu)
                    
                    if self.config.get("algo") == "greedy_search":
                        action_chosen = action
                        obs_chosen = obs_simu
                        min_rho = float(obs_simu.rho.max())
                        break
            except Exception as e:
                logger.error(f"Simulation failed for action at substation {substation_id}: {e}")

        if rho_list and self.config.get("algo") != "greedy_search":
            sorted_indices = np.argsort(rho_list)
            rho_list = [rho_list[i] for i in sorted_indices]
            list_action = [list_action[i] for i in sorted_indices]
            list_nb_modif_objects = [list_nb_modif_objects[i] for i in sorted_indices]
            obs_list = [obs_list[i] for i in sorted_indices]
            
            min_rho = rho_list[0]
            action_chosen = list_action[0]
            obs_chosen = obs_list[0]

        return action_chosen, obs_chosen, min_rho, rho_list, list_action, list_nb_modif_objects, obs_list

    def get_topology_action(
        self,
        observation: ObservationIDF2023,
        base_action: BaseAction,
        action_space_n1: List[BaseAction],
        action_space_overload: List[BaseAction],
        min_rho: Optional[float],
    ) -> Tuple[
        Optional[BaseAction],
        Optional[ObservationIDF2023],
        List[float],
        List[BaseAction],
        List[int],
        List[ObservationIDF2023],
    ]:
        """
        Retrieves the best topology-changing action based on the observation and available actions.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            base_action (BaseAction): Base action from the agent.
            action_space_n1 (List[BaseAction]): N-1 action space.
            action_space_overload (List[BaseAction]): Overload action space.
            min_rho (Optional[float]): Minimum rho value to beat.

        Returns:
            Tuple containing:
                - action_chosen (Optional[BaseAction]): The selected action.
                - obs_chosen (Optional[ObservationIDF2023]): The observation corresponding to the chosen action.
                - rho_list (List[float]): List of rho values from simulations.
                - list_action (List[BaseAction]): List of actions tried.
                - list_nb_modif_objects (List[int]): Number of modified objects for each action.
                - obs_list (List[ObservationIDF2023]): List of simulated observations.
        """
        if min_rho is None:
            min_rho = float(observation.rho.max())

        # Try recovering reference topology first
        action_chosen, obs_chosen, min_rho = self.recover_reference_topology(
            observation, base_action, min_rho
        )
        if action_chosen is not None and obs_chosen and obs_chosen.rho.max() < self.config.get("rho_safe", 0.9):
            logger.info("Successfully recovered reference topology")
            return action_chosen, obs_chosen, [min_rho], [action_chosen], \
                   [action_chosen.as_dict()["set_bus_vect"]["nb_modif_objects"]], [obs_chosen]

        # If recovery fails, try changing substation topology
        action_chosen, obs_chosen, min_rho, rho_list, list_action, list_nb_modif_objects, obs_list = \
            self.change_substation_topology(
                observation, base_action, action_space_n1, action_space_overload, min_rho
            )

        return action_chosen, obs_chosen, rho_list, list_action, list_nb_modif_objects, obs_list

    def act(
        self,
        observation: ObservationIDF2023,
        act: BaseAction,
        action_space_n1: List[BaseAction],
        action_space_overload: List[BaseAction],
        reward: Optional[float] = None,
        done: bool = False,
    ) -> BaseAction:
        """
        Decides on an action based on the current observation and action space.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            act (BaseAction): Current action from the agent.
            action_space_n1 (List[BaseAction]): N-1 action space.
            action_space_overload (List[BaseAction]): Overload action space.
            reward (Optional[float], optional): Reward received from the previous action.
            done (bool, optional): Indicates if the episode is done.

        Returns:
            BaseAction: The chosen topology-changing action.
        """
        if done:
            logger.info("Episode done, returning do-nothing action")
            return self.do_nothing

        # Get topology changing action
        topo_act, topo_obs, _, _, _, _ = self.get_topology_action(
            observation,
            act,
            action_space_n1,
            action_space_overload,
            min_rho=self.config.get("min_rho"),
        )

        if topo_act is not None:
            logger.info("Selected topology-changing action")
            return topo_act
        else:
            logger.info("No suitable topology action found, returning do-nothing action")
            return self.do_nothing