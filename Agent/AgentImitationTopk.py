#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AgentImitationTopk Module
=========================
This module implements the Imitation class, which handles loading configurations,
initializing models, loading checkpoints, making predictions, and evaluating model
performance for imitation learning in power grid dispatching.

It leverages PyTorch and PyTorch Geometric for model operations and interacts
with the Grid2Op environment to process observations and predict optimal actions.
"""

import copy
import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

try:
    from Imitation_model.src.model import GraphTransformerModel, PowerGridModel, GATModel
except ModuleNotFoundError:
    from .Imitation_model.src.model import GraphTransformerModel, PowerGridModel, GATModel
    


# Set CUDA device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

# Configure logger for this module
logger = logging.getLogger(__name__)


def get_graph_sub(obs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract node features, edges, and edge attributes from the energy graph of the observation.

    Parameters
    ----------
    obs : ObservationIDF2023
        The observation object containing the energy graph.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing node features, edges, and edge attributes as numpy arrays.
    """
    # Extract graph data from the observation
    graph = obs.get_energy_graph()

    # Node features, edges, and edge attributes
    node_features = [list(graph.nodes[n].values()) for n in graph.nodes]
    edges = list(graph.edges)
    edge_attrs = [list(graph.edges[e].values()) for e in graph.edges]

    return (
        np.array(node_features, dtype=np.float16),
        np.array(edges),
        np.array(edge_attrs, dtype=np.float16)
    )


def convert_obs_to_data(obs) -> Data:
    """
    Convert an observation into a PyTorch Geometric Data object.

    Parameters
    ----------
    obs : ObservationIDF2023
        The observation object to be converted.

    Returns
    -------
    Data
        A PyTorch Geometric Data object containing node features, edge indices, and edge attributes.
    """
    # Convert observation to graph components
    node_features, edges, edge_attrs = get_graph_sub(obs)

    # Create PyTorch tensors for node features, edge indices, and edge attributes
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Return PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data


class Imitation:
    """
    Imitation Learning Model for Power Grid Dispatching.

    This class handles loading configurations, initializing models, loading checkpoints,
    making predictions, and evaluating model performance.

    Attributes
    ----------
    config : dict
        Configuration settings loaded from a YAML file.
    mapping_dict : dict
        Dictionary mapping actions to their corresponding indices.
    inverse_mapping_dict : dict
        Inverse mapping dictionary for translating indices back to actions.
    n_class : int
        Number of output classes/actions.
    device : torch.device
        Device to run the model on (CPU or CUDA).
    checkpoint_path : str
        Path to the model checkpoint file.
    model : torch.nn.Module
        The initialized PyTorch model.
    """

    def __init__(
        self,
        config_path: str,
        model_class,
        mapping_dict_action: dict
    ):
        """
        Initialize the Imitation Learning model with configuration, model class, and mapping dictionary.

        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file.
        model_class : class
            Model class (GraphTransformerModel, PowerGridModel, or GATModel).
        mapping_dict_action : dict
            Action mapping dictionary.
        """
        # Load configuration and mapping dictionary
        self.config = self.load_config(config_path)
        self.config["architecture"] = (
            "PowerGridModel" if model_class == PowerGridModel else
            "GAT" if model_class == GATModel else
            "GraphTransformer"
        )
        self.mapping_dict = mapping_dict_action
        self.inverse_mapping_dict = {v: k for k, v in self.mapping_dict.items()}
        self.n_class = len(self.mapping_dict)

        # Initialize device and model checkpoint path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = (
            self.config["model_PowerGrid"] if model_class == PowerGridModel else
            self.config["model_GAT"] if model_class == GATModel else
            self.config["model_GraphTransformer"]
        )

        # Initialize and load model
        self.model = self.initialize_model(model_class, self.n_class)
        self.load_checkpoint(self.checkpoint_path)
        self.model.to(self.device)

    def load_config(self, config_path: str) -> dict:
        """
        Load the configuration from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file.

        Returns
        -------
        dict
            Loaded configuration dictionary.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def initialize_model(self, model_class, n_class: int) -> torch.nn.Module:
        """
        Initialize the model based on the specified model class.

        Parameters
        ----------
        model_class : class
            Model class (GraphTransformerModel, PowerGridModel, or GATModel).
        n_class : int
            Number of output classes.

        Returns
        -------
        torch.nn.Module
            Initialized model object.
        """
        if model_class == GATModel:
            model = model_class(
                input_features=self.config["num_feature_node"],
                hidden_units=self.config["hidden_units"],
                num_heads=self.config["num_heads"],
                num_classes=n_class
            )
        elif model_class == PowerGridModel:
            model = PowerGridModel(
                input_features=self.config["num_feature_node"],
                hidden_units=256,  # Hidden units for PowerGridModel
                num_heads=8,        # Number of heads for PowerGridModel
                num_classes=n_class
            )
        elif model_class == GraphTransformerModel:
            model = GraphTransformerModel(
                dim=self.config["dim"],
                depth=self.config["depth"],
                num_classes=n_class,
                num_feature_node=self.config["num_feature_node"],
                num_feature_edge=self.config["num_feature_edge"],
                device=self.device
            )
        else:
            raise ValueError(
                "Unsupported architecture. Choose 'GraphTransformer', 'GAT', or 'PowerGridModel'."
            )
        return model

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict_actions(self, dataset_sample: Data, topk: int = 10) -> np.ndarray:
        """
        Predict the top-k actions based on the input dataset sample.

        Parameters
        ----------
        dataset_sample : Data
            PyTorch Geometric Data object representing the observation.
        topk : int, optional
            Number of top predictions to return, by default 10.

        Returns
        -------
        np.ndarray
            Top-k predicted action indices.
        """
        dataset_test = [dataset_sample]
        loader_test = DataLoader(
            dataset_test, batch_size=1, shuffle=False, collate_fn=Batch.from_data_list
        )

        for sample in loader_test:
            sample = sample.to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(sample)
                _, preds = outputs.topk(topk, dim=1, largest=True, sorted=True)
            return preds.cpu().numpy()[0]

    def predict_actions_id(
        self,
        dataset_sample: Data,
        topk: int = 10,
        order: bool = False
    ) -> List[str]:
        """
        Predict action IDs based on the input dataset sample.

        Parameters
        ----------
        dataset_sample : Data
            PyTorch Geometric Data object representing the observation.
        topk : int, optional
            Number of top predictions to return, by default 10.
        order : bool, optional
            Whether to return ordered action IDs, by default False.

        Returns
        -------
        List[str]
            Top-k predicted action IDs.
        """
        preds = self.predict_actions(dataset_sample, topk)
        preds = sorted(preds) if order else preds
        id_actions = [
            self.inverse_mapping_dict.get(pred, "Unknown") for pred in preds
        ]
        return id_actions

    def calculate_topk_accuracy(
        self,
        preds: np.ndarray,
        label: torch.Tensor,
        topk: Tuple[int, ...] = (1, 5, 10)
    ) -> dict:
        """
        Calculate top-k accuracy for the predictions.

        Parameters
        ----------
        preds : np.ndarray
            Predicted action indices.
        label : torch.Tensor
            True label.
        topk : Tuple[int, ...], optional
            Tuple of k values to calculate accuracy for, by default (1, 5, 10).

        Returns
        -------
        dict
            Dictionary of top-k accuracies.
        """
        accuracies = {}
        label = label.cpu().numpy()[0]  # Ensure label is an integer
        for k in topk:
            correct = label in preds[:k]
            accuracies[f'top{k}'] = 1 if correct else 0
        return accuracies

    def evaluate_dataset(
        self,
        dataset: List[Data],
        topk: Tuple[int, ...] = (1, 5, 10)
    ) -> dict:
        """
        Evaluate the dataset and calculate average top-k accuracies.

        Parameters
        ----------
        dataset : List[Data]
            List of PyTorch Geometric Data objects representing the dataset.
        topk : Tuple[int, ...], optional
            Tuple of k values to calculate accuracy for, by default (1, 5, 10).

        Returns
        -------
        dict
            Dictionary of average top-k accuracies.
        """
        total_accuracies = {f'top{k}': 0 for k in topk}
        total_samples = len(dataset)

        for sample in dataset:
            preds = self.predict_actions(sample, topk=max(topk))
            accuracies = self.calculate_topk_accuracy(
                preds, sample.y, topk=topk
            )
            for k in topk:
                total_accuracies[f'top{k}'] += accuracies[f'top{k}']

        # Calculate the average accuracy
        average_accuracies = {
            f'top{k}': total_accuracies[f'top{k}'] / total_samples
            for k in topk
        }
        return average_accuracies

    def predict_from_obs(self, obs, topk: int = 10) -> List[str]:
        """
        Predict the classification from an observation.

        Parameters
        ----------
        obs : ObservationIDF2023
            The observation object to be classified.
        topk : int, optional
            The number of top predictions to return, by default 10.

        Returns
        -------
        List[str]
            A list of predicted action IDs.
        """
        data = convert_obs_to_data(obs)
        return self.predict_actions_id(data, topk=topk)


if __name__ == "__main__":
    import grid2op
    import datasets
    from lightsim2grid import LightSimBackend
    from Imitation_model.src.Data_processing import convert_data, CustomDataset

    def load_mapping_dict(mapping_dict_path: str) -> dict:
        """
        Load action mapping dictionary from a JSON file.

        Parameters
        ----------
        mapping_dict_path : str
            Path to the mapping dictionary file.

        Returns
        -------
        dict
            Loaded mapping dictionary.
        """
        with open(mapping_dict_path, 'r') as json_file:
            mapping_dict = json.load(json_file)
        return mapping_dict

    # Load action mapping dictionaries for Overload and N1 cases
    action_space_overload = load_mapping_dict("../ACTION_REDUCED/action_dict_Overload.json")
    action_space_n1 = load_mapping_dict("../ACTION_REDUCED/action_dict_N1.json")

    # Initialize Imitation Learning models for N-1 and Overload cases
    imitation_N1 = Imitation(
        "./Imitation_model/config/config_N1.yaml",
        GraphTransformerModel,
        action_space_n1
    )
    imitation_Overload = Imitation(
        "./Imitation_model/config/config_Overload.yaml",
        PowerGridModel,
        action_space_overload
    )

    # Load the Grid2Op environment
    env = grid2op.make(
        "/home/van.tuan/DATA/EVAL_L2RPN2023/Baseline/input_data_local",
        backend=LightSimBackend()
    )

    # Take an initial observation from the environment
    obs, _, done, _ = env.step(env.action_space({}))

    # Predict actions for N-1 and Overload cases
    print("Prediction from observation in N1 case:", imitation_N1.predict_from_obs(obs, topk=10))
    print("Prediction from observation in Overload case:", imitation_Overload.predict_from_obs(obs, topk=10))

    # Load test datasets for N-1 and Overload cases
    dataset_N1 = datasets.load_dataset('Lajavaness/IEEE-118_attacked_line_test')["test"]
    dataset_overload = datasets.load_dataset('Lajavaness/IEEE-118_overload_test')["test"]

    # Convert datasets
    test_ds_N1 = convert_data(dataset_N1)
    test_ds_overload = convert_data(dataset_overload)

    # Initialize custom datasets
    dataset_test_N1 = CustomDataset(
        test_ds_N1[0], test_ds_N1[1], test_ds_N1[2], test_ds_N1[3]
    )
    dataset_test_overload = CustomDataset(
        test_ds_overload[0], test_ds_overload[1], test_ds_overload[2], test_ds_overload[3]
    )

    # Evaluate N-1 case
    total_accuracies_N1 = {f'top{k}': 0 for k in (1, 5, 10, 15, 20)}
    for sample in dataset_test_N1:
        preds = imitation_N1.predict_actions(sample, topk=20)
        accuracies = imitation_N1.calculate_topk_accuracy(
            preds, sample.y, topk=(1, 5, 10, 15, 20)
        )
        for k in (1, 5, 10, 15, 20):
            total_accuracies_N1[f'top{k}'] += accuracies[f'top{k}']
    avg_acc_N1 = {
        f'top{k}': total_accuracies_N1[f'top{k}'] / len(dataset_test_N1)
        for k in (1, 5, 10, 15, 20)
    }
    print("Average Accuracies in N-1 case:", avg_acc_N1)

    # Evaluate Overload case
    total_accuracies_Overload = {f'top{k}': 0 for k in (1, 5, 10, 15, 20)}
    for sample in dataset_test_overload:
        preds = imitation_Overload.predict_actions(sample, topk=20)
        accuracies = imitation_Overload.calculate_topk_accuracy(
            preds, sample.y, topk=(1, 5, 10, 15, 20)
        )
        for k in (1, 5, 10, 15, 20):
            total_accuracies_Overload[f'top{k}'] += accuracies[f'top{k}']
    avg_acc_Overload = {
        f'top{k}': total_accuracies_Overload[f'top{k}'] / len(dataset_test_overload)
        for k in (1, 5, 10, 15, 20)
    }
    print("Average Accuracies in Overload case:", avg_acc_Overload)