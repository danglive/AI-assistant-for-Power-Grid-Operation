#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AgentReconnection Module
========================
This module implements the AgentReconnection class, which is responsible for reconnecting power lines
that have been disconnected. The agent ensures that the reconnection process minimizes the load (rho)
on the grid and supports both area-based reconnection and global reconnection of lines.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation

# Setup logger for this module
logger = logging.getLogger(__name__)


class AgentReconnection(BaseAgent):
    """
    AgentReconnection is responsible for reconnecting power lines that have been disconnected,
    ensuring the reconnection process minimizes the load (rho) on the grid.

    It supports both area-based reconnection and global reconnection of lines.

    Args:
        env (grid2op.Environment.BaseEnv): The Grid2Op environment.
        action_space (grid2op.Action.ActionSpace): The action space of the environment.
        config (dict): Configuration settings.
        time_step (int, optional): Time step for the simulation of actions. Defaults to 1.
        verbose (int, optional): Verbosity level for logging. Defaults to 1.

    Attributes:
        env (grid2op.Environment.BaseEnv): The Grid2Op environment.
        action_space (grid2op.Action.ActionSpace): The action space of the environment.
        lines_in_area (List[List[int]]): List of line IDs grouped by area.
        area (bool): If True, reconnection will be area-based.
        time_step (int): Time step for the simulation of actions.
        verbose (int): Verbosity level for logging.
        do_nothing (grid2op.Action.BaseAction): Action that does nothing.
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
        self.action_space = action_space
        self.lines_in_area: List[List[int]] = [
            list_ids for list_ids in env._game_rules.legal_action.lines_id_by_area.values()
        ]
        self.area: bool = config.get("areas", False)  # If True, reconnection will be area-based
        self.time_step = time_step
        self.verbose = verbose
        self.do_nothing = action_space({})  # Default action that does nothing

    def recon_line_area(
        self, observation: BaseObservation
    ) -> Tuple[List[BaseAction], List[Optional[BaseObservation]]]:
        """
        Reconnects lines on a per-area basis, ensuring that each area's reconnection
        minimizes the load (rho) in that specific area.

        Args:
            observation (BaseObservation): The current observation of the grid.

        Returns:
            Tuple[List[BaseAction], List[Optional[BaseObservation]]]:
                - action_chosen_per_area: List of actions to reconnect lines in each area.
                - obs_simu_chosen_per_area: List of simulated observations after applying the reconnection actions.
        """
        line_status = observation.line_status  # Status of the lines (True: connected, False: disconnected)
        cooldown = observation.time_before_cooldown_line  # Cooldown time before lines can be reconnected
        can_be_reco = ~line_status & (cooldown == 0)  # Lines that can be reconnected

        num_areas = len(self.lines_in_area)
        min_rho_per_area = [np.inf] * num_areas
        action_chosen_per_area: List[BaseAction] = [self.do_nothing] * num_areas
        obs_simu_chosen_per_area: List[Optional[BaseObservation]] = [None] * num_areas

        if np.any(can_be_reco):
            reconnectable_line_ids = np.where(can_be_reco)[0]
            actions_and_ids = [
                (self.action_space({"set_line_status": [(line_id, +1)]}), line_id)
                for line_id in reconnectable_line_ids
            ]

            for action, line_id in actions_and_ids:
                try:
                    obs_simu, _, _, info = observation.simulate(action, time_step=self.time_step)
                    if info.get('exception', False):
                        logger.warning(f"Simulation exception for line {line_id}: {info.get('exception')}")
                        continue

                    # Find which area this line belongs to
                    area_id = next(
                        (idx for idx, lines in enumerate(self.lines_in_area) if line_id in lines),
                        None,
                    )

                    if area_id is not None and obs_simu.rho.max() < min_rho_per_area[area_id]:
                        # Update the best action and observation for this area
                        action_chosen_per_area[area_id] = action
                        obs_simu_chosen_per_area[area_id] = obs_simu
                        min_rho_per_area[area_id] = obs_simu.rho.max()
                        logger.debug(
                            f"Area {area_id}: Reconnected line {line_id} with rho {obs_simu.rho.max()}"
                        )
                except Exception as e:
                    logger.error(f"Error simulating reconnection for line {line_id}: {e}", exc_info=True)

        return action_chosen_per_area, obs_simu_chosen_per_area

    def combine_actions(
        self, base_action: BaseAction, list_of_actions: List[BaseAction]
    ) -> BaseAction:
        """
        Combine multiple actions into one.

        Args:
            base_action (BaseAction): The initial action to combine with other actions.
            list_of_actions (List[BaseAction]): List of actions to combine.

        Returns:
            BaseAction: The final combined action.
        """
        combined_action = base_action
        for action in list_of_actions:
            combined_action += action
        return combined_action

    def reco_line(
        self, observation: BaseObservation
    ) -> Tuple[Optional[BaseAction], Optional[BaseObservation]]:
        """
        Reconnect lines globally, without considering areas, ensuring that the reconnection
        minimizes the overall load (rho) on the grid.

        Args:
            observation (BaseObservation): The current observation of the grid.

        Returns:
            Tuple[Optional[BaseAction], Optional[BaseObservation]]:
                - action_chosen: The chosen action to reconnect the line with the least load.
                - obs_simu_chosen: The simulated observation after applying the reconnection action.
        """
        line_status = observation.line_status  # Status of the lines
        cooldown = observation.time_before_cooldown_line  # Cooldown time for lines
        can_be_reco = ~line_status & (cooldown == 0)  # Lines that can be reconnected

        min_rho = np.inf
        action_chosen: Optional[BaseAction] = None
        obs_simu_chosen: Optional[BaseObservation] = None

        if np.any(can_be_reco):
            reconnectable_line_ids = np.where(can_be_reco)[0]
            actions = [
                self.action_space({"set_line_status": [(line_id, +1)]})
                for line_id in reconnectable_line_ids
            ]

            for action, line_id in zip(actions, reconnectable_line_ids):
                try:
                    obs_simu, _, _, info = observation.simulate(action, time_step=self.time_step)
                    if info.get('exception', False):
                        logger.warning(f"Simulation exception for line {line_id}: {info.get('exception')}")
                        continue

                    if obs_simu.rho.max() < min_rho:
                        action_chosen = action
                        obs_simu_chosen = obs_simu
                        min_rho = obs_simu.rho.max()
                        logger.debug(f"Reconnected line {line_id} with rho {min_rho}")
                except Exception as e:
                    logger.error(f"Error simulating reconnection for line {line_id}: {e}", exc_info=True)

        return action_chosen, obs_simu_chosen

    def act(
        self,
        observation: BaseObservation,
        act: BaseAction,
        reward: Optional[float] = None,
        done: bool = False,
    ) -> BaseAction:
        """
        Decide the best reconnection action based on the current observation.
        This method checks whether to perform area-based or global reconnections.

        Args:
            observation (BaseObservation): The current observation of the grid.
            act (BaseAction): The current action being built (to combine with reconnection actions).
            reward (Optional[float], optional): Reward from the environment. Defaults to None.
            done (bool, optional): Indicates if the episode is finished. Defaults to False.

        Returns:
            BaseAction: The final action after possibly adding reconnection actions.
        """
        if done:
            logger.info("Episode done. Returning current action.")
            return act

        if self.area:
            logger.debug("Performing area-based reconnection.")
            reco_actions, _ = self.recon_line_area(observation)
            act = self.combine_actions(act, reco_actions)
        else:
            logger.debug("Performing global reconnection.")
            reco_action, _ = self.reco_line(observation)
            if reco_action is not None:
                act += reco_action

        return act