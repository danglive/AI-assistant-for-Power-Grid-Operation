import os
import ast
import torch
import datasets
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


# Function to remove unnecessary columns from the dataset
def remove_col(dataset):
    """
    Remove specific unnamed columns from the dataset, if present.
    
    Parameters:
    - dataset: The dataset from which columns will be removed.
    
    Returns:
    - dataset: The dataset with specified columns removed.
    """
    columns_to_remove = ['Unnamed: 0.1', 'Unnamed: 0']
    existing_columns = [col for col in columns_to_remove if col in dataset.column_names]
    if existing_columns:
        dataset = dataset.remove_columns(existing_columns)
    return dataset


# Function to load datasets based on the config
def load_datasets(config):
    """
    Load training, validation, and test datasets based on configuration.
    
    Parameters:
    - config: A dictionary containing dataset configuration such as paths and usage flags.
    
    Returns:
    - tuple: A tuple of datasets (train, validation, test).
    """
    if config["use_local_data"]:
        dataset = datasets.load_dataset('csv', data_files={
            'train': config["train_data"],
            'validation': config["valid_data"],
            'test': config["test_data"]
        })
    else:
        dataset = datasets.load_dataset(config["dataset_path"])
    
    # Process datasets to remove unnecessary columns
    train_dataset = remove_col(dataset["train"]) if "train" in dataset.keys() else None
    valid_dataset = remove_col(dataset["validation"]) if "validation" in dataset.keys() else None
    test_dataset = remove_col(dataset["test"]) if "test" in dataset.keys() else None
    
    return train_dataset, valid_dataset, test_dataset


# Function to convert dataset into lists of node features, edge indices, edge attributes, and labels
def convert_data(data):
    """
    Convert the dataset into node features, edge indices, edge attributes, and labels.
    
    Parameters:
    - data: The dataset containing node features, edge attributes, and edge indices.
    
    Returns:
    - tuple: Lists of node features, edge indices, edge attributes, and labels.
    """
    labels = data["label"]

    # Extract edge attributes
    if "edge_attr_sub" in data.column_names:
        edge_attr = [ast.literal_eval(etr) for etr in tqdm(data["edge_attr_sub"], desc="Processing edge attributes")]
    elif "edge_attr" in data.column_names:
        edge_attr = [ast.literal_eval(etr) for etr in tqdm(data["edge_attr"], desc="Processing edge attributes")]
    else:
        edge_attr = []

    # Extract node features
    if "node_features_sub" in data.column_names:
        node_features = [ast.literal_eval(nf) for nf in tqdm(data["node_features_sub"], desc="Processing node features")]
    elif "node_features" in data.column_names:
        node_features = [ast.literal_eval(nf) for nf in tqdm(data["node_features"], desc="Processing node features")]
    else:
        node_features = []

    # Extract edge indices
    if "edge_index_sub" in data.column_names:
        edge_indices = [ast.literal_eval(ei) for ei in tqdm(data["edge_index_sub"], desc="Processing edge indices")]
    elif "edge_index" in data.column_names:
        edge_indices = [ast.literal_eval(ei) for ei in tqdm(data["edge_index"], desc="Processing edge indices")]
    else:
        edge_indices = []

    return node_features, edge_indices, edge_attr, labels


# Custom dataset class to wrap the node features, edge indices, edge attributes, and labels
class CustomDataset:
    def __init__(self, node_features, edge_indices, edge_attr, labels):
        """
        Initialize the CustomDataset class with node features, edge indices, edge attributes, and labels.
        
        Parameters:
        - node_features: A list of node features for each graph.
        - edge_indices: A list of edge indices for each graph.
        - edge_attr: A list of edge attributes for each graph.
        - labels: A list of labels corresponding to each graph.
        """
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.edge_attr = edge_attr
        self.labels = labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get the sample at index `idx` from the dataset.
        
        Parameters:
        - idx: Index of the sample to retrieve.
        
        Returns:
        - Data: A PyTorch Geometric `Data` object containing node features, edge indices, edge attributes, and the label.
        """
        node_feature = torch.tensor(self.node_features[idx], dtype=torch.float)
        edge_index = torch.tensor(self.edge_indices[idx], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(self.edge_attr[idx], dtype=torch.float)
        label = torch.tensor([self.labels[idx]], dtype=torch.long)
        return Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, y=label)