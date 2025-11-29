import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from cosmic import *


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 1
        output_dim = 10
        hidden_dim = 64
        layers = 8
        self.radius = 2
        self.eps = self.radius

        self.lift = Lift2D()
        self.cosmo_layers = nn.ModuleList(
            [
                NeuralFieldCosmo(
                    in_channels=hidden_dim if i > 0 else input_dim,
                    out_channels=hidden_dim if i < layers - 1 else output_dim,
                    dim=2,
                    radius=self.radius,
                )
                for i in range(layers)
            ]
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def train_step(self, Xy, data):
        logits, _, _ = self.forward(data)
        y = data.y.long()
        loss = self.criterion(logits, y)
        return loss, logits

    def eval_step(self, Xy, data):
        return self.train_step(Xy, data)

    def forward(self, data):
        features = data.atom_features.float()
        pos = data.atom_pos.float()
        edge_index = gnn.radius_graph(pos, self.eps, data.vertex2molecule)
        L = self.lift(features, pos, edge_index, data.vertex2molecule)
        features = L.features
        for layer in self.cosmo_layers:
            features = layer(L.source, L.target, features, L.hood_coords)
        mol_features, max_indices = scatter_max(
            features, L.lifted2inst, dim_size=data.num_molecules, dim=0
        )
        return mol_features, mol_features, max_indices


class Beta2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 7
        output_dim = 1
        hidden_dim = 32
        layers = 2
        self.radius = 1.54

        self.lift = Lift2D()
        self.cosmo_layers = nn.ModuleList(
            [
                NeuralFieldCosmo(
                    in_channels=hidden_dim if i > 0 else input_dim,
                    out_channels=hidden_dim if i < layers - 1 else output_dim,
                    dim=2,
                    radius=self.radius,
                    field_activation=nn.Tanh,
                )
                for i in range(layers)
            ]
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def train_step(self, Xy, data):
        logits, _, _ = self.forward(data)
        y = data.y.float()
        loss = self.criterion(logits.squeeze(-1), y)
        return loss, logits

    def eval_step(self, Xy, data):
        return self.train_step(Xy, data)

    def forward(self, data):
        features = data.atom_features.float()
        pos = data.atom_pos.float()
        edge_index = data.molecule_edges.T
        L = self.lift(features, pos, edge_index, data.vertex2molecule)
        features = L.features
        for layer in self.cosmo_layers:
            features = layer(L.source, L.target, features, L.hood_coords)
        mol_features, max_indices = scatter_max(
            features, L.lifted2inst, dim_size=data.num_molecules, dim=0
        )
        return mol_features, mol_features, max_indices


class Beta3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 7
        output_dim = 1
        hidden_dim = 32
        layers = 2
        self.radius = 1.54

        self.lift = Lift3D()
        self.cosmo_layers = nn.ModuleList(
            [
                NeuralFieldCosmo(
                    in_channels=hidden_dim if i > 0 else input_dim,
                    out_channels=hidden_dim if i < layers - 1 else output_dim,
                    dim=3,
                    radius=self.radius,
                )
                for i in range(layers)
            ]
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def train_step(self, Xy, data):
        logits, _, _ = self.forward(data)
        y = data.y.float()
        loss = self.criterion(logits.squeeze(-1), y)
        return loss, logits

    def eval_step(self, Xy, data):
        return self.train_step(Xy, data)

    def forward(self, data):
        features = data.atom_features.float()
        pos = data.atom_pos.float()
        edge_index = data.molecule_edges.T
        L = self.lift(features, pos, edge_index, data.vertex2molecule)
        features = L.features
        for layer in self.cosmo_layers:
            features = layer(L.source, L.target, features, L.hood_coords)
        mol_features, max_indices = scatter_max(
            features, L.lifted2inst, dim_size=data.num_molecules, dim=0
        )
        return mol_features, mol_features, max_indices


class QM9Model(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 5
        output_dim = 1
        hidden_dim = 512
        mlp_dim = 128
        layers = 6
        self.radius = 1.54
        self.k = 4

        self.lift = Lift3D()
        self.cosmo_layers = nn.ModuleList(
            [
                PointTransformerCosmo(
                    in_channels=hidden_dim if i > 0 else input_dim,
                    out_channels=hidden_dim,
                    dim=3,
                    radius=self.radius,
                )
                for i in range(layers)
            ]
        )
        self.cosmo_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for i in range(layers)]
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim),
        )
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[36_000, 38_000, 40_000], gamma=0.5
        )

    def train_step(self, Xy, data):
        logits, _, _ = self.forward(data)
        y = data.y.float()
        logits = logits.squeeze(-1)
        loss = self.criterion(logits, y)
        return loss, logits

    def eval_step(self, Xy, data):
        return self.train_step(Xy, data)

    def forward(self, data):
        features = data.atom_features.float()
        pos = data.atom_pos.float()
        edge_index = gnn.knn_graph(pos, self.k, data.vertex2molecule)
        L = self.lift(features, pos, edge_index, data.vertex2molecule)
        features = L.features
        all_features = []
        for layer, norm in zip(self.cosmo_layers, self.cosmo_norms):
            features = layer(L.source, L.target, features, L.hood_coords)
            features = norm(features)
            all_features.append(features.clone())
        all_features = torch.stack(all_features, dim=1).mean(dim=1)
        mol_features = scatter_max(
            all_features, L.lifted2inst, dim_size=data.num_molecules, dim=0
        )[0]
        logits = self.mlp(mol_features)
        torch.cuda.empty_cache()
        return logits, None, None
