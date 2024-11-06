import os
import ast
import torch
import wandb
import torch.nn.functional as F
import datasets
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv, GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from datetime import datetime
from .GraphTransformer import GraphTransformer


# GraphTransformerModel class defines the structure of a graph-based Transformer model
class GraphTransformerModel(nn.Module):
    def __init__(
        self, dim, depth, num_classes, num_feature_node, num_feature_edge,
        edge_dim=None, with_feedforwards=True, gated_residual=True, 
        rel_pos_emb=True, device=None, dropout_rate=0.5
    ):
        super(GraphTransformerModel, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_dim = edge_dim if edge_dim else dim

        # Initialize the encoder and feature encoders
        self.encoder = GraphTransformer(
            dim=dim, depth=depth, edge_dim=edge_dim, 
            with_feedforwards=with_feedforwards, 
            gated_residual=gated_residual, 
            rel_pos_emb=rel_pos_emb).to(self.device)
        
        # Encoders for node and edge features
        self.node_feature_encoder = nn.Linear(num_feature_node, dim).to(self.device)
        self.edge_feature_encoder = nn.Linear(num_feature_edge, edge_dim).to(self.device)

        # Prediction heads that output class logits
        self.pred_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=dropout_rate),
                nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=dropout_rate),
                nn.Linear(dim, num_classes)
            ).to(self.device) for _ in range(1)
        ])

    def encode_features(self, batch):
        """Encodes node and edge features into tensors for further processing."""
        z_x = self.node_feature_encoder(batch.x.float())
        z_e = self.edge_feature_encoder(batch.edge_attr.float())
        nodes, mask = to_dense_batch(z_x, batch.batch)
        edges = to_dense_adj(batch.edge_index, batch.batch, edge_attr=z_e)
        return nodes, edges, mask

    def forward(self, batch):
        """Performs forward pass through the model."""
        nodes, edges, mask = self.encode_features(batch)
        nodes, edges = self.encoder(nodes, edges, mask=mask)
        res = scatter_mean(nodes[mask], batch.batch, dim=0)
        preds = [head(res) for head in self.pred_heads]
        return torch.cat(preds, dim=-1)


# PowerGridModel defines a multi-layer GNN with TransformerConv layers
class PowerGridModel(nn.Module):
    def __init__(self, input_features, hidden_units, num_heads, num_classes):
        super(PowerGridModel, self).__init__()
        
        # Three GNN layers using TransformerConv
        self.gat1 = TransformerConv(input_features, hidden_units, heads=num_heads, edge_dim=26, concat=True)
        self.gat2 = TransformerConv(hidden_units * num_heads, hidden_units, heads=num_heads, edge_dim=26, concat=True)
        self.gat3 = TransformerConv(hidden_units * num_heads, hidden_units, heads=num_heads, edge_dim=26, concat=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_units * num_heads, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        """Performs forward pass through the GNN model."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Apply the three layers of TransformerConv and activation
        x = self.relu(self.gat1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.relu(self.gat2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.relu(self.gat3(x, edge_index, edge_attr))

        # Perform global pooling and pass through final FC layer
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.fc(x)


# GATModel is a multi-layer GNN using GATv2Conv layers
class GATModel(nn.Module):
    def __init__(self, input_features, hidden_units, num_heads, num_classes):
        super(GATModel, self).__init__()
        
        # Two GNN layers using GATv2Conv
        self.gat1 = GATv2Conv(input_features, hidden_units, heads=num_heads, edge_dim=26, concat=True)
        self.gat2 = GATv2Conv(hidden_units * num_heads, hidden_units, heads=num_heads, edge_dim=26, concat=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_units * num_heads, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        """Performs forward pass through the GNN model."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Apply the two layers of GATv2Conv and activation
        x = self.relu(self.gat1(x, edge_index, edge_attr))
        x = self.relu(self.gat2(x, edge_index, edge_attr))
        
        # Perform global pooling and pass through final FC layer
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.fc(x)


# FocalLoss function is useful for handling imbalanced datasets
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Calculates Focal Loss between inputs and targets."""
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


# Evaluation function to calculate loss and top-k accuracies
def evaluate(loader, net, loss_fnc, calc_loss=False, topk=(1,)):
    total_loss, correct, total = 0, {k: 0 for k in topk}, 0
    net.eval()

    with torch.no_grad():
        for data in loader:
            data = data.to('cuda')
            outputs = net(data)
            
            # Optionally calculate loss
            if calc_loss:
                loss = loss_fnc(outputs, data.y)
                total_loss += loss.item() * data.y.size(0)

            # Calculate top-k accuracies
            _, preds = outputs.topk(max(topk), 1, True, True)
            correct_preds = preds.eq(data.y.view(-1, 1).expand_as(preds))
            for k in topk:
                correct[k] += correct_preds[:, :k].sum().item()
            total += data.y.size(0)

    # Return average loss and accuracies
    loss = total_loss / total if calc_loss else None
    accuracies = {k: correct[k] / total for k in topk}
    return loss, accuracies


# Training loop function
def train(
    net, loader_train, loader_validation, save_best_model, model_save_dir, 
    num_epochs, learning_rate, weight_decay, early_stop_threshold, 
    loss_function, config
):
    wandb.init(
        project=config['wandb_project'],
        entity=config.get('wandb_entity', None),
        config=config
    )

    # Select loss function (FocalLoss or CrossEntropyLoss)
    loss_fnc = FocalLoss() if loss_function == 'FocalLoss' else nn.CrossEntropyLoss()

    # Set optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.5)

    # Initialize lists to track losses and accuracies
    train_losses, validation_losses, train_accuracies, validation_accuracies = [], [], [], []
    best_validation_loss, epochs_no_improve = float('inf'), 0

    # Create directory to save models
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # Save path for best model
    model_save_path = os.path.join(
        model_save_dir, f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )

    for epoch in range(num_epochs):
        if epochs_no_improve >= early_stop_threshold:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

        # Training phase
        net.train()
        train_loss = 0
        for data in tqdm(loader_train, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            data = data.to('cuda')
            optimizer.zero_grad()
            out = net(data)
            loss = loss_fnc(out, data.y)
            train_loss += loss.item() * data.y.size(0)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Calculate average training loss
        train_loss /= len(loader_train.dataset)
        train_losses.append(train_loss)

        # Validation phase
        validation_loss, acc_validation = evaluate(
            loader_validation, net, loss_fnc, calc_loss=True, topk=(1, 5, 10)
        )
        validation_losses.append(validation_loss)
        _, acc_train = evaluate(loader_train, net, loss_fnc, calc_loss=False, topk=(1, 5, 10))
        train_accuracies.append(acc_train[1])
        validation_accuracies.append(acc_validation[1])

        # Log results to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'train_acc_top1': acc_train[1],
            'validation_acc_top1': acc_validation[1],
            'validation_acc_top5': acc_validation[5],
            'validation_acc_top10': acc_validation[10]
        })

        # Save the best model based on validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            if save_best_model:
                torch.save(net.state_dict(), model_save_path)
                print(f"Saved better model with validation loss: {validation_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch: {epoch} -- Train Loss: {train_loss:.4f} -- Val Loss: {validation_loss:.4f} || "
            f"Train Acc Top-1: {acc_train[1]:.4f} -- Val Acc Top-1: {acc_validation[1]:.4f} -- "
            f"Top-5: {acc_validation[5]:.4f} -- Top-10: {acc_validation[10]:.4f}"
        )

    # Finish logging in W&B
    wandb.finish()

    return train_losses, validation_losses, train_accuracies, validation_accuracies