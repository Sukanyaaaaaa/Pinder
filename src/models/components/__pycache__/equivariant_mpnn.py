import rootutils
import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing,global_mean_pool
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch
# setup root dir and pythonpath
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.pinder_dataset import PinderDataset
from src.models.components.utils import (
    compute_euler_angles_from_rotation_matrices,
    compute_rotation_matrix_from_ortho6d,
    get_R
)



class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, out_dim=128, aggr="add"):
        r"""Message Passing Neural Network Layer

        This layer is equivariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension d
            edge_dim: (int) - edge feature dimension d_e
            aggr: (str) - aggregation function \oplus (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim

        #
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + 1, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
        )

        self.mlp_pos = Sequential(
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), Linear(emb_dim, 1)
        )  # MLP \psi
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
        )  # MLP \phi
        # ===========================================
         
        self.lin_out = Linear(emb_dim, out_dim)
        
        


    def forward(self, data):
        """
        The forward pass updates node features h via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: [(n, d),(n,3)] - updated node features
        """

        #
        h, pos, edge_index = data
        h_out, pos_out = self.propagate(edge_index=edge_index, h=h, pos=pos)
        h_out = self.lin_out(h_out)
        return h_out, pos_out, edge_index
        # ==========================================

    #
    def message(self, h_i, h_j, pos_i, pos_j):
        # Compute distance between nodes i and j (Euclidean distance)
        # distance_ij = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)  # (e, 1)
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)

        # Concatenate node features, edge features, and distance
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        pos_diff = pos_diff * self.mlp_pos(msg)  # (e, 2d + d_e + 1)

        # (e, d)
        return msg, pos_diff

    #   ...
    #
    def aggregate(self, inputs, index):
        """The aggregate function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages m_ij from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in input

        Returns:
            aggr_out: (n, d) - aggregated messages m_i
        """
        msgs, pos_diffs = inputs

        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)

        pos_aggr = scatter(pos_diffs, index, dim=self.node_dim, reduce="mean")

        return msg_aggr, pos_aggr

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr = aggr_out

        upd_out = self.mlp_upd(torch.cat((h, msg_aggr), dim=-1))

        upd_pos = pos + pos_aggr

        return upd_out, upd_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"


class PinderMPNNModel(Module):
    def __init__(self, input_dim=1, emb_dim=64, num_heads=2):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations (the constituent MPNN layers
        are equivariant to 3D rotations and translations).

        Args:
            emb_dim: (int) - hidden dimension d
            input_dim: (int) - initial node feature dimension d_n
            edge_dim: (int) - edge feature dimension d_e
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        # Linear projection for initial node features
        self.lin_in_rec = Linear(input_dim, emb_dim)
        self.lin_in_lig = Linear(input_dim, emb_dim)

        # Stack of MPNN layers
        self.receptor_mpnn = Sequential(
            EquivariantMPNNLayer(emb_dim, 128, aggr="mean"),
            EquivariantMPNNLayer(128, 256, aggr="mean"),
            EquivariantMPNNLayer(256, 512, aggr="mean")
            # EquivariantMPNNLayer(512, 512, aggr="mean"),
        )
        self.ligand_mpnn = Sequential(
            EquivariantMPNNLayer(64, 128, aggr="mean"),
            EquivariantMPNNLayer(128, 256, aggr="mean"),
            EquivariantMPNNLayer(256, 512, aggr="mean")
            # EquivariantMPNNLayer(512, 512, aggr="mean"),
        )

        # Cross-attention layer
        self.rec_cross_attention = nn.MultiheadAttention(128, num_heads, batch_first=True)
        self.lig_cross_attention = nn.MultiheadAttention(128, num_heads, batch_first=True)
        # self.rec_cross_attention = GraphMultiheadAttention(256, 256, num_heads)
        # self.lig_cross_attention = GraphMultiheadAttention(256, 256, num_heads)
        
        # MLPs for translation prediction
        self.fc_translation_rec = nn.Linear(128 , 3)
        self.fc_translation_lig = nn.Linear(128 , 3)
        
        self.fc_rotation_rec = nn.Linear(128 , 9)
        self.fc_rotation_lig = nn.Linear(128 , 9)
        
        self.pool = global_mean_pool
        
    def forward(self, batch):
        """
        The main forward pass of the model.

        Args:
            batch: Same as in forward_rot_trans.

        Returns:
            transformed_ligands: List of tensors, each of shape (1, num_ligand_atoms, 3)
            representing the transformed ligand coordinates after applying the predicted
            rotation and translation.
        """
        h_receptor = self.lin_in_rec(batch["receptor"].x)
        h_ligand = self.lin_in_lig(batch["ligand"].x)

        pos_receptor = batch["receptor"].pos
        pos_ligand = batch["ligand"].pos

        h_receptor, pos_receptor, _ = self.receptor_mpnn(
            (h_receptor, pos_receptor, batch["receptor", "receptor"].edge_index)
        )

        h_ligand, pos_ligand, _ = self.ligand_mpnn(
            (h_ligand, pos_ligand, batch["ligand", "ligand"].edge_index)
        )


#         attn_output_rec, _ = self.rec_cross_attention(h_receptor, h_ligand, h_ligand)
        
#         attn_output_rec, _ = self.rec_cross_attention(h_receptor, h_ligand, h_ligand)
        
#         attn_output_rec, _ = self.rec_cross_attention(h_receptor, h_ligand, h_ligand)
#         attn_output_rec = self.pool(attn_output_rec , batch['receptor'].batch)
       
#         attn_output_lig, _ = self.lig_cross_attention(h_ligand, h_receptor, h_receptor)
#         attn_output_lig = self.pool(attn_output_lig , batch['ligand'].batch)
#         # attn_output_rec, _ = self.rec_cross_attention(receptor_nodes,h_receptor, batch["receptor", "receptor"].edge_index, batch["batch"])
#         # attn_output_rec = self.pool(attn_output_rec, batch['batch'])
#         # attn_output_lig, _ = self.lig_cross_attention(ligand_nodes, h_ligand, batch["ligand", "ligand"].edge_index, batch["batch"])
#         # attn_output_lig = self.pool(attn_output_lig, batch['batch'])
        
#         rotation_matrix_rec = self.fc_rotation_rec(attn_output_rec)
#         rotation_matrix_rec = rotation_matrix_rec.view(-1, 3, 3)
        
#         rotation_matrix_lig = self.fc_rotation_lig(attn_output_lig)
#         rotation_matrix_lig = rotation_matrix_lig.view(-1, 3, 3)
        
#         translation_vector_rec = self.fc_translation_rec(attn_output_rec)
#         translation_vector_lig = self.fc_translation_lig(attn_output_lig)
        
#         receptor_coords = torch.zeros_like(batch['receptor'].pos, dtype=batch['receptor'].pos.dtype)
#         for i in range(rotation_matrix_rec.size(0)):  # Loop over batch
#             mask = batch['receptor'].batch == i 
#             # print(receptor_coords[mask])# Mask for the current batch
#             receptor_coords[mask] = torch.matmul(
#                                      batch['receptor'].pos[mask].to(rotation_matrix_rec.dtype), rotation_matrix_rec[i]) + translation_vector_rec[i].to(batch['receptor'].pos.dtype)
#             # print(receptor_coords[mask])
# # Transform ligand coordinates
#         ligand_coords = torch.zeros_like(batch['ligand'].pos, dtype=batch['ligand'].pos.dtype)
#         for i in range(rotation_matrix_lig.size(0)):  # Loop over batch
#             mask = batch['ligand'].batch == i  # Mask for the current batch
#             ligand_coords[mask] = torch.matmul(
#                                   batch['ligand'].pos[mask].to(rotation_matrix_lig.dtype), rotation_matrix_lig[i]) + translation_vector_lig[i].to(batch['ligand'].pos.dtype)

        return pos_receptor, pos_ligand


if __name__ == "__main__":
    file_paths = ["/home/sukanya/pinder_challenge/pinder_challenge-1/data/processed/apo/test/1a19__A1_P11540--1a19__B1_P11540.pt", "/home/sukanya/pinder_challenge/pinder_challenge-1/data/processed/apo/test/1ag9__A1_P61949--1ag9__B1_P61949.pt","/home/sukanya/pinder_challenge/pinder_challenge-1/data/processed/apo/test/1ht8__A1_P05979--1ht8__B1_P05979.pt","/home/sukanya/pinder_challenge/pinder_challenge-1/data/processed/apo/test/1rve__A1_P04390--1rve__B1_P04390.pt"]
    dataset = PinderDataset(file_paths=file_paths)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    # batch = next(iter(loader))
    model = PinderMPNNModel()
    # print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    # receptor_coords, ligand_coords = model(loader)
    for batch in loader:
    # Pass a batch to the model
         receptor_coords, ligand_coords = model(batch)
         print(receptor_coords)
         print(ligand_coords)
