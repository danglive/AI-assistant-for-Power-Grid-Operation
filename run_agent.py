#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PowerGridAgent Module
=====================
This module contains the implementation of the PowerGridAgent class, which integrates multiple
decision-making agents for controlling grid topology and handling reconnections, redispatching,
and overload management using either Greedy Search or Imitation Learning.
"""

import os
import zlib
import json
import base64
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional, List
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from grid2op.Action import BaseAction, ActionSpace
from grid2op.l2rpn_utils.idf_2023 import ObservationIDF2023
from Agent.AgentRecoverTopo import AgentRecoverTopo
from Agent.AgentReconnection import AgentReconnection
from Agent.AgentTopology import AgentTopology
from Agent.DispatcherAgent import DispatcherAgent
from Agent.AgentImitationTopk import Imitation
from Agent.Imitation_model.src.model import GraphTransformerModel, PowerGridModel

# Setup logger for this module
logger = logging.getLogger(__name__)

class EncodedTopologyAction:
    """
    A class to handle topology (set_bus) actions encoded as a zlib-compressed base64 string.
    This facilitates easier storage and retrieval of actions from CSV files.
    """

    def __init__(self, action: Optional[BaseAction]):
        """Encode the action into a compressed string upon initialization."""
        self.data: str = self.encode_action(action)

    def to_action(self, env: BaseEnv) -> BaseAction:
        """Decode the stored string back into a Grid2Op action."""
        return self.decode_action(self.data, env)

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return self.data

    @staticmethod
    def encode_action(action: Optional[BaseAction]) -> str:
        """
        Encode a set_bus action into a base64 string for easy storage.
        If no action is provided, returns "0" representing the "do nothing" action.
        """
        if not action:
            return "0"

        # Ensure we are only encoding set_bus actions
        assert (
            not action._modif_inj
            and not action._modif_change_bus
            and not action._modif_set_status
            and not action._modif_change_status
            and not action._modif_redispatch
            and not action._modif_storage
            and not action._modif_curtailment
            and not action._modif_alarm
        ), "Given action type can only encode set_bus actions."

        # Special case: No topology action -> Encode as "0"
        if not action._modif_set_bus:
            return "0"

        # Compress the topology action
        packed_action = zlib.compress(action._set_topo_vect, level=1)
        encoded_action = base64.b64encode(packed_action)
        return encoded_action.decode("utf-8")

    @staticmethod
    def decode_action(act_string: str, env: BaseEnv) -> BaseAction:
        """
        Decode the encoded string back into a Grid2Op action.
        """
        unpacked_act: BaseAction = env.action_space()

        if act_string == "0":
            return unpacked_act  # "Do nothing" action

        decoded = base64.b64decode(act_string.encode("utf-8"))
        unpacked = np.frombuffer(zlib.decompress(decoded), dtype=np.int32)
        unpacked_act._set_topo_vect = unpacked
        unpacked_act._modif_set_bus = True
        return unpacked_act


class PowerGridAgent(BaseAgent):
    """
    PowerGridAgent integrates multiple decision-making agents for controlling grid topology and
    handling reconnections, redispatching, and overload management using either Greedy Search or
    Imitation Learning.

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
        rho_danger (float): Threshold for dangerous grid state.
        rho_safe (float): Threshold for safe grid state.
        action_space_n1 (List[BaseAction]): N-1 action space.
        action_space_overload (List[BaseAction]): Overload action space.
        action_space_n1_id (dict): Mapping of action IDs for N-1.
        action_space_overload_id (dict): Mapping of action IDs for overload.
        imitation_N1 (Optional[Imitation]): Imitation Learning model for N-1.
        imitation_Overload (Optional[Imitation]): Imitation Learning model for overload.
        agent_topology (AgentTopology): Agent for topology control.
        line_reconnection (AgentReconnection): Agent for line reconnection.
        dispatcher (DispatcherAgent): Agent for redispatching.
        agent_recover_topo (AgentRecoverTopo): Agent for recovering topology.
    """

    def __init__(
        self,
        env: BaseEnv,
        action_space: ActionSpace,
        config: dict,
        time_step: int = 1,
        verbose: int = 1,
    ):
        super().__init__(action_space)
        self.env = env
        self.do_nothing = action_space({})  # Action that does nothing
        self.config = config
        self.rho_danger = float(self.config.get("rho_danger", 0.99))
        self.rho_safe = float(self.config.get("rho_safe", 0.9))
        self.env = env
        
        # Load the N-1 and overload action spaces
        action_space_path_N1 = self.config.get("action_space_path_N1")
        action_space_path_Overload = self.config.get("action_space_path_Overload")

        # Load the N-1 and overload action spaces
        self._load_topo_actions(action_space_path_N1, action_space_path_Overload)
        # Load IDs of the N-1 and overload actions
        self._load_mapping_dict(action_space_path_N1, action_space_path_Overload)

        # Initialize Imitation Learning Models for N-1 and Overload
        self.imitation_N1 = (
            Imitation(
                self.config["imitation_n1_config_path"], 
                GraphTransformerModel, 
                self.action_space_n1_id,
            )
            if self.config["imitation"] else None
        )

        self.imitation_Overload = (
            Imitation(
                self.config["imitation_overload_config_path"],  
                PowerGridModel, 
                self.action_space_overload_id,
            )
            if self.config["imitation"] else None
        )

        # Initialize sub-agents
        self.agent_topology = AgentTopology(
            env, action_space, config, time_step, verbose
        )
        self.line_reconnection = AgentReconnection(
            env, action_space, config, time_step, verbose
        )
        self.dispatcher = DispatcherAgent(
            env, action_space, config, time_step, verbose
        )
        self.agent_recover_topo = AgentRecoverTopo(env)

    def _load_mapping_dict(self, action_space_path_N1: str, action_space_path_Overload: str) -> None:
        """
        Load action mapping dictionaries from JSON files.

        Args:
            action_space_path_N1 (str): Path to the N-1 action space JSON file.
            action_space_path_Overload (str): Path to the overload action space JSON file.
        """
        try:
            with open(action_space_path_N1, "r") as json_file:
                self.action_space_n1_id = json.load(json_file)
        except Exception as e:
            logger.error(f"Error loading N-1 action mapping: {e}")
            self.action_space_n1_id = {}

        try:
            with open(action_space_path_Overload, "r") as json_file:
                self.action_space_overload_id = json.load(json_file)
        except Exception as e:
            logger.error(f"Error loading overload action mapping: {e}")
            self.action_space_overload_id = {}

    def _load_topo_actions(self, action_space_path_n1: str, action_space_path_overload: str) -> None:
        """
        Load topology actions from the provided file paths for N-1 and overload cases.

        Args:
            action_space_path_n1 (str): Path to the N-1 action space JSON file.
            action_space_path_overload (str): Path to the overload action space JSON file.
        """
        logger.info(f"Loading N-1 actions from {action_space_path_n1}")
        logger.info(f"Loading Overload actions from {action_space_path_overload}")

        self.action_space_n1 = self._load_action_to_grid2op(action_space_path_n1)
        self.num_action_n1 = len(self.action_space_n1)

        self.action_space_overload = self._load_action_to_grid2op(action_space_path_overload)
        self.num_action_overload = len(self.action_space_overload)

        logger.info(f"Loaded {self.num_action_n1} N-1 actions")
        logger.info(f"Loaded {self.num_action_overload} Overload actions")

    def _load_action_to_grid2op(self, action_id_path: str) -> List[BaseAction]:
        """
        Load action IDs from a JSON file and convert them into Grid2Op actions.

        Args:
            action_id_path (str): Path to the action ID JSON file.

        Returns:
            List[BaseAction]: List of Grid2Op actions.
        """
        if not action_id_path or not os.path.exists(action_id_path):
            logger.error(f"Action ID path '{action_id_path}' is invalid.")
            return []

        if not action_id_path.endswith(".json"):
            logger.error(f"Action ID path '{action_id_path}' is not a JSON file.")
            return []

        try:
            with open(action_id_path, "r") as json_file:
                action_ids = list(json.load(json_file).keys())
        except Exception as e:
            logger.error(f"Error reading action IDs from '{action_id_path}': {e}")
            return []

        all_actions = []
        for action_id in action_ids:
            try:
                action = EncodedTopologyAction.decode_action(action_id, self.env)
                all_actions.append(action)
            except Exception as e:
                logger.error(f"Error decoding action ID '{action_id}': {e}")
        return all_actions

    def _load_topk_actions(self, observation: ObservationIDF2023, topk: int):
        """
        Use Imitation Learning to predict the top-k actions for both N-1 and overload cases.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            topk (int): Number of top actions to predict.

        Returns:
            Tuple[List[BaseAction], List[BaseAction]]: Predicted actions for N-1 and overload cases.
        """
        # Predict top-k actions for N-1
        topk_action_n1 = self.imitation_N1.predict_from_obs(observation, topk=topk)
        self.action_space_n1 = [
            EncodedTopologyAction.decode_action(action_id, self.env)
            for action_id in topk_action_n1
        ]
        self.num_action_n1 = len(self.action_space_n1)

        # Predict top-k actions for overload
        topk_action_overload = self.imitation_Overload.predict_from_obs(
            observation, topk=topk
        )
        self.action_space_overload = [
            EncodedTopologyAction.decode_action(action_id, self.env)
            for action_id in topk_action_overload
        ]
        self.num_action_overload = len(self.action_space_overload)

        logger.info(f"Predicted {self.num_action_n1} N-1 actions using Imitation")
        logger.info(f"Predicted {self.num_action_overload} Overload actions using Imitation")

        return self.action_space_n1, self.action_space_overload

    def _handle_safe_rho(self, observation: ObservationIDF2023) -> BaseAction:
        """
        Handle the case where the grid is in a safe state and we can recover the previous topology.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.

        Returns:
            BaseAction: Action to recover the previous topology.
        """
        return self.agent_recover_topo.act(observation)

    def _reset_dispatcher(self) -> None:
        """Reset the dispatcher's flow computations and error state."""
        self.dispatcher.flow_computed[:] = np.NaN
        self.dispatcher._prev_por_error.value[:] = 0.0

    def _update_prev_por_error(self, observation: ObservationIDF2023, prev_ok: np.ndarray) -> None:
        """
        Update the dispatcher's previous error state based on the new observation.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            prev_ok (np.ndarray): Boolean array indicating valid previous flow computations.
        """
        self.dispatcher._prev_por_error.value[prev_ok] = np.minimum(
            self.dispatcher.flow_computed[prev_ok] - observation.p_or[prev_ok], 0.0
        )
        self.dispatcher._prev_por_error.value[~prev_ok] = 0.0

    def _simulate_initial_action(
        self, observation: ObservationIDF2023, action: BaseAction
    ) -> Optional[ObservationIDF2023]:
        """
        Simulate the initial action to check for any potential errors.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            action (BaseAction): Action to simulate.

        Returns:
            Optional[ObservationIDF2023]: Simulated observation if successful, else None.
        """
        try:
            simulated_obs, reward, done, info = observation.simulate(action, time_step=1)
            if not info.get("exception", False):
                return simulated_obs
            else:
                logger.warning(f"Simulation resulted in an exception: {info.get('exception')}")
        except Exception as e:
            logger.error(f"Simulation failed with exception: {e}")
        return None

    def _get_maneuver_from_parade(self, action: BaseAction) -> pd.DataFrame:
        """
        Extract the maneuver details from an action's dictionary representation.

        Args:
            action (BaseAction): The action to extract maneuvers from.

        Returns:
            pd.DataFrame: DataFrame containing maneuver details.
        """
        df = pd.DataFrame.from_dict(action.as_dict()["set_bus_vect"][action.as_dict()["set_bus_vect"]["modif_subs_id"][0]], orient='index').reset_index()
        df.columns = ['Connection', 'Type', 'ID element', 'Assign Bus']
        df['At substation'] = action.as_dict()["set_bus_vect"]["modif_subs_id"]*len(df)
        return df

    def _get_affected_elements(self, observation: ObservationIDF2023) -> dict:
        """
        Get information about the elements affected by overloads.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.

        Returns:
            dict: Dictionary containing overload lines, their loads, and associated substations.
        """
        overload_lines = self.agent_topology._get_ranked_overloads(observation)
        overload_loads = [observation.rho[line] for line in overload_lines]
        substations_ex = [observation.line_ex_to_subid[line] for line in overload_lines]
        substations_or = [observation.line_or_to_subid[line] for line in overload_lines]
        return {
            "overload line": overload_lines,
            "max load line": overload_loads,
            "substations at line extremity": substations_ex,
            "substations at line origin": substations_or,
        }

    def _simulate_future_steps(self, actions: List[BaseAction], observation: ObservationIDF2023) -> pd.DataFrame:
        """
        Simulate the next 12 steps for multiple actions to evaluate their effectiveness.

        Args:
            actions (List[BaseAction]): List of actions to simulate.
            observation (ObservationIDF2023): Current observation from the environment.

        Returns:
            pd.DataFrame: DataFrame containing rho values over time for each action.
        """
        time_steps = range(1, 13)  # Simulate next 12 steps
        simulation_results = []
        for action in actions[: self.config.get("multiple_parades", 5)]:
            rho_values = []
            for step in time_steps:
                try:
                    sim_obs, _, _, _ = observation.simulate(action, time_step=step)
                    rho_values.append(sim_obs.rho.max())
                except Exception as e:
                    logger.error(f"Simulation failed at step {step} with exception: {e}")
                    rho_values.append(np.nan)
            simulation_results.append({
                "Parade": EncodedTopologyAction(action).data,
                **{f"t+{step*5}": rho for step, rho in zip(time_steps, rho_values)}
            })
        df = pd.DataFrame(simulation_results)
        return df

    def act(
        self, observation: ObservationIDF2023, reward: Optional[float] = None, done: bool = False
    ) -> BaseAction:
        """
        The main decision-making function that returns the best action for the given observation.

        Args:
            observation (ObservationIDF2023): Current observation from the environment.
            reward (float, optional): Reward received from the previous action.
            done (bool, optional): Indicates if the episode is done.

        Returns:
            BaseAction: The chosen action to apply.
        """
        if done:
            return self.do_nothing  # If the episode is done, return "do nothing"

        if observation.current_step == 0:
            self._reset_dispatcher()  # Reset dispatcher on first step

        # Update the previous error for redispatch decisions
        prev_ok = np.isfinite(self.dispatcher.flow_computed)
        self._update_prev_por_error(observation, prev_ok)

        # Start with the default "do nothing" action and simulate the initial action
        action = self.action_space()
        simulated_obs = self._simulate_initial_action(observation, action)

        # Check reconnection actions
        action = self.line_reconnection.act(observation, action)

        # Dispatcher will check for redispatching or energy storage needs
        self.dispatcher.flow_computed[:] = np.NaN

        max_rho = round(float(observation.rho.max()), 4)

        # Handle overloaded grid (rho exceeds danger threshold)
        if max_rho > self.rho_danger:
            if self.config.get("imitation", False):
                logger.info("Using Imitation Learning to select top-k topology actions.")
                action_space_n1, action_space_overload = self._load_topk_actions(
                    observation, self.config.get("topk", 5)
                )
            else:
                logger.info("Using Greedy Search to select actions from the entire action space.")
                action_space_n1 = self.action_space_n1
                action_space_overload = self.action_space_overload

            # Get the best topology action
            (
                topo_action,
                topo_obs,
                rho_list,
                list_action,
                list_nb_modif_objects,
                obs_list,
            ) = self.agent_topology.get_topology_action(
                observation, action, action_space_n1, action_space_overload, self.config.get("min_rho")
            )

            if topo_action:
                action += topo_action
                simulated_obs = topo_obs
            else:
                if self.config.get("dispatching", False):
                    # Update the dispatcher with the new observation
                    self.dispatcher._update_storage_power_obs(observation)
                    self.dispatcher.update_parameters(simulated_obs or observation, safe=False)
                    # Perform redispatch or energy storage if needed
                    action = self.dispatcher.act(observation, action)

        # Handle safe grid (rho below safe threshold)
        elif max_rho < self.rho_safe:
            action = self._handle_safe_rho(observation)

        # If grid is in a normal state, do nothing
        else:
            action += self.do_nothing

        return action


# Main execution block
if __name__ == "__main__":
    import sys
    import pandas as pd
    import warnings
    import grid2op
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from lightsim2grid import LightSimBackend
    from grid2op.PlotGrid import PlotMatplot

    """
    This block is for testing and demonstrating the PowerGridAgent's decision-making process.
    It simulates the agent's actions in a Grid2Op environment and displays relevant information.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load the configuration file
    config_file = "config.json"
    with open(config_file, "r") as cfg_file:
        CONFIG = json.load(cfg_file)

    print(f"PowerGridAgent with config: {CONFIG}")

    # Initialize environment with LightSimBackend
    env_name = CONFIG.get("env_name", "./Scenarios_test")
    env = grid2op.make(env_name, backend=LightSimBackend())

    # Initialize PowerGridAgent with relevant configurations
    agent = PowerGridAgent(
        env=env,
        action_space=env.action_space,
        config=CONFIG,
        verbose=True,
    )

    # Load test configuration and retrieve seeds for each scenario
    test_config_file = os.path.join(env_name, "config_test.json")
    with open(test_config_file, "r") as f:
        test_config = json.load(f)

    # Set environment seeds based on chronic data
    env_seeds = [
        int(test_config["episodes_info"][os.path.split(el)[-1]]["seed"])
        for el in sorted(env.chronics_handler.real_data.subpaths)
    ]
    scenarios = env.chronics_handler.subpaths[:5]
    seeds = {scen: int(seed) for scen, seed in zip(scenarios, env_seeds)}
    print(f"Scenarios: {scenarios}")
    print(f"Seeds: {seeds}")

    # Set up the first scenario
    scen_id = scenarios[0]
    print(f"Setting up scenario: {scen_id}")
    env.set_id(scen_id)
    env.seed(seeds[scen_id])
    obs = env.reset()
    agent.reset(obs)
    done = False
    info = None
    act = None  # Initialize act for the first step
    plot_helper = PlotMatplot(env.observation_space)
    _ = plot_helper.plot_obs(obs)

    # Simulation loop
    for nb_step in tqdm(range(obs.max_step)):
        #fig = plot_helper.plot_obs(obs, line_info="rho", load_info=None, gen_info=None, storage_info=None)
        #fig.savefig(f"figure/step_{nb_step+1}")
        prev_obs = obs
        prev_act = act
        prev_info = info

        # Plot the evolution of power grid maximum load (rho)
        plt.figure()
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, obs.max_step)
        ax.set_ylim(0.5, 2)
        ax.plot(nb_step, obs.rho.max(), 'ro')
        ax.set_xlabel("Step")
        ax.set_ylabel("rho.max()")
        ax.set_title("Evolution of rho.max() Over Time")
        plt.show()

        # Reset dispatcher on the first step
        if obs.current_step == 0:
            agent._reset_dispatcher()

        # Update previous error for redispatch decisions
        prev_ok = np.isfinite(agent.dispatcher.flow_computed)
        agent._update_prev_por_error(obs, prev_ok)

        # Simulate initial "do nothing" action
        act = env.action_space({})
        _obs = agent._simulate_initial_action(obs, act)

        # Handle reconnection actions if applicable
        act = agent.line_reconnection.act(obs, act)
        if "set_line_status" in act.as_dict():
            print(
                f"STREAMLIT BOX: Reconnect line: {act.as_dict()['set_line_status']['connected_id']}"
            )
        else:
            print("STREAMLIT BOX: Reconnect line: None")

        # Reset dispatcher for redispatching or energy storage
        agent.dispatcher.flow_computed[:] = np.NaN

        # Define grid states
        grid_states = ["Danger", "Under Safe", "Safe"]
        max_rho = round(float(obs.rho.max()), 4)

        # Handle under safe grid state (rho below safe threshold)
        if max_rho < agent.rho_safe:
            print(f"STREAMLIT BOX: Power grid state: {grid_states[2]}")
            act = agent._handle_safe_rho(obs)
            if "set_bus_vect" in act.as_dict():
                maneuvers = agent._get_maneuver_from_parade(act)
                print(
                    f"STREAMLIT BOX: Recover Topology Configuration, Parades suggested to Operator:\n{maneuvers}"
                )

        # Handle grid overloads (rho exceeds danger threshold)
        elif max_rho > agent.rho_danger:
            print(f"STREAMLIT BOX: Power grid state: {grid_states[0]}")

            # Get overloaded lines and relevant data
            df_affected = pd.DataFrame(agent._get_affected_elements(obs))
            print(f"STREAMLIT BOX: Affected Elements:\n{df_affected}")

            # Load top actions for N-1 and overload situations
            action_space_n1, action_space_overload = agent._load_topk_actions(
                obs, agent.config.get("topk", 5)
            )

            # Determine the best topology actions based on grid state
            (
                topo_act,
                topo_obs,
                rho_list,
                list_action,
                list_nb_modif_objects,
                obs_list,
            ) = agent.agent_topology.get_topology_action(
                obs, act, action_space_n1, action_space_overload, None
            )

            # Check if topology action is found
            if list_action:
                # Handle multiple suggested actions
                df_rho_future = agent._simulate_future_steps(list_action, obs)
                print(
                    f"STREAMLIT BOX: Information of Load simulated by Grid2Op:\n{df_rho_future}"
                )
                for idx, parade in enumerate(
                    list_action[: agent.config.get("multiple_parades", 5)]
                ):
                    maneuvers = agent._get_maneuver_from_parade(parade)
                    print(
                        f"STREAMLIT BOX: Parade {idx} with {list_nb_modif_objects[idx]} maneuvers and load after apply: {rho_list[idx]}:\n{maneuvers}"
                    )

                # Simulate operator selection (Here we just pick the first one)
                print("STREAMLIT BOX: Operator selects Parade 0 to apply")
                idx = 0  # Simulate user input here
                topo_act = list_action[idx]
                topo_obs = obs_list[idx]
                act += topo_act
                _obs = topo_obs
            else:
                # If solution topology is not found then try dispatching
                # Update dispatcher based on new observations
                agent.dispatcher._update_storage_power_obs(obs)
                agent.dispatcher.update_parameters(_obs or obs, safe=False)
                act = agent.dispatcher.act(obs, act)

                # Manage redispatch or energy storage if necessary
                dispatching = {}
                act_dict = act.as_dict()
                if "redispatch" in act_dict:
                    dispatching["redispatch"] = [list(act_dict["redispatch"])]
                if "storage_power" in act_dict:
                    dispatching["storage_power"] = [list(act_dict["storage_power"])]
                if "curtailment" in act_dict:
                    dispatching["curtailment"] = [list(act_dict["curtailment"])]
                if dispatching:
                    df_dispatching = pd.DataFrame(dispatching)
                    print(
                        f"STREAMLIT BOX: Parade not found, dispatching suggested\n{df_dispatching}"
                    )
                else:
                    print("STREAMLIT BOX: Parade not found, dispatching suggested\nNone")
        # If grid is in a normal state, do nothing
        else:
            print(f"STREAMLIT BOX: Power grid state: {grid_states[1]}")
            print("STREAMLIT BOX: Do nothing")
            act += agent.do_nothing

        # Step through the environment with the chosen action
        obs, reward, done, info = env.step(act)

        # Handle blackout conditions
        if done and (nb_step != prev_obs.max_step - 1):
            logger.error(f"Grid is blackout with cause:\n{info}")
            break

    print("Simulation completed.")