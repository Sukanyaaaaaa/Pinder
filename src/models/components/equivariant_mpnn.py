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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BigBirdMultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, global_tokens, num_random=3, window_size=3):
        """
        BigBird-style Multi-Head Cross-Attention for variable-length sequences.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            global_tokens (list): Indices of tokens with global attention.
            num_random (int): Number of random tokens each token attends to.
            window_size (int): Size of sliding window for attention.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.global_tokens = global_tokens
        self.num_random = num_random
        self.window_size = window_size

        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def create_sparse_attention_mask(self, seq_len):
        """
        Dynamically create a BigBird-style sparse attention mask for variable sequence lengths.

        Args:
            seq_len (int): Length of the sequence.

        Returns:
            torch.Tensor: Sparse attention mask of shape (seq_len, seq_len).
        """
        mask = torch.zeros(seq_len, seq_len)

        # Global attention
        for token in self.global_tokens:
            if token < seq_len:
                mask[token, :] = 1  # Full attention for global tokens
                mask[:, token] = 1

        # Sliding window attention
        for i in range(seq_len):
            for j in range(max(0, i - self.window_size), min(seq_len, i + self.window_size + 1)):
                mask[i, j] = 1

        # Random attention
        for i in range(seq_len):
            random_indices = np.random.choice(seq_len, min(self.num_random, seq_len), replace=False)
            mask[i, random_indices] = 1

        return mask

    def forward(self, query, key, value):
        """
        Forward pass for BigBird-style multi-head cross-attention.

        Args:
            query (Tensor): Query tensor of shape (N, embed_dim).
            key (Tensor): Key tensor of shape (N, embed_dim).
            value (Tensor): Value tensor of shape (N, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (N, embed_dim).
        """
        print('quer', query , query.shape)
        seq_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim, "Embedding dimensions must match"

        # Linear projections
        Q = self.query_proj(query).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.key_proj(key).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.value_proj(value).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # Compute scaled dot-product attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply sparse attention mask
        sparse_mask = self.create_sparse_attention_mask(seq_len).to(attn_logits.device)
        attn_logits = attn_logits.masked_fill(sparse_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)

        # Weighted sum of values
        attention = torch.matmul(attn_weights, V)
        attention = attention.transpose(0, 1).contiguous().view(seq_len, self.embed_dim)

        # Final linear projection
        output = self.output_proj(attention)

        return output


class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=32, out_dim=32, aggr="add"):
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
    def __init__(self, input_dim=1, emb_dim=64, num_heads=1):
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
            EquivariantMPNNLayer(emb_dim, 64, aggr="mean"),
            EquivariantMPNNLayer(64, 64, aggr="mean"),
            # EquivariantMPNNLayer(64, 64, aggr="mean")
            # EquivariantMPNNLayer(512, 512, aggr="mean"),
        )
        self.ligand_mpnn = Sequential(
            EquivariantMPNNLayer(emb_dim, 64, aggr="mean"),
            EquivariantMPNNLayer(64, 64, aggr="mean"),
            # EquivariantMPNNLayer(64, 64, aggr="mean")
            # EquivariantMPNNLayer(512, 512, aggr="mean"),
        )

        # # Cross-attention layer
        self.rec_cross_attention = nn.MultiheadAttention(64, num_heads, batch_first=True)
        self.lig_cross_attention = nn.MultiheadAttention(64, num_heads, batch_first=True)
        # self.rec_cross_attention = BigBirdMultiHeadCrossAttention(256, num_heads , global_tokens= [0])
        # self.lig_cross_attention = BigBirdMultiHeadCrossAttention(256, num_heads , global_tokens= [0])
        
        # MLPs for translation prediction
        self.fc_translation_rec = nn.Linear(64 , 3)
        self.fc_translation_lig = nn.Linear(64 , 3)
        
        self.fc_rotation_rec = nn.Linear(64 , 9)
        self.fc_rotation_lig = nn.Linear(64, 9)
        
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
        # print(batch)
        h_receptor = self.lin_in_rec(batch["receptor"].x)
        h_ligand = self.lin_in_lig(batch["ligand"].x)

        pos_receptor = batch["receptor"].pos
        pos_ligand = batch["ligand"].pos

        h_receptor, pos_receptor, _ = self.receptor_mpnn(
            (h_receptor, pos_receptor, batch["receptor", "receptor"].edge_index)
        )
        # print(h_receptor ,h_receptor.shape , "before")
        # h_receptor = self.pool(h_receptor , batch['receptor'].batch)
        # print(batch['receptor'].batch , "later")
        h_ligand, pos_ligand, _ = self.ligand_mpnn(
            (h_ligand, pos_ligand, batch["ligand", "ligand"].edge_index)
        )
        # h_receptor = self.pool(h_receptor , batch['receptor'].batch)
        # h_ligand = self.pool(h_ligand , batch['ligand'].batch)
        h_receptor = h_receptor.mean(dim=0, keepdim=True)
        h_ligand = h_ligand.mean(dim=0, keepdim=True)
        
# # #         print('h_ligadn', h_ligand.shape)

        attn_output_rec, _ = self.rec_cross_attention(h_receptor, h_ligand, h_ligand)
        
# # #         attn_output_rec, _ = self.rec_cross_attention(h_receptor, h_ligand, h_ligand)
        
# # #         attn_output_rec, _ = self.rec_cross_attention(h_receptor, h_ligand, h_ligand)
#         # attn_output_rec = self.pool(attn_output_rec , batch['receptor'].batch)
       
        attn_output_lig, _ = self.lig_cross_attention(h_ligand, h_receptor, h_receptor)
# # #         attn_output_lig = self.pool(attn_output_lig , batch['ligand'].batch)
# # #         # attn_output_rec, _ = self.rec_cross_attention(receptor_nodes,h_receptor, batch["receptor", "receptor"].edge_index, batch["batch"])
# # #         # attn_output_rec = self.pool(attn_output_rec, batch['batch'])
# # #         # attn_output_lig, _ = self.lig_cross_attention(ligand_nodes, h_ligand, batch["ligand", "ligand"].edge_index, batch["batch"])
#         # attn_output_lig = self.pool(attn_output_lig, batch['ligand'].batch)
        
        rotation_matrix_rec = self.fc_rotation_rec(attn_output_rec)
        rotation_matrix_rec = rotation_matrix_rec.view(-1, 3, 3)
        
        rotation_matrix_lig = self.fc_rotation_lig(attn_output_lig)
        rotation_matrix_lig = rotation_matrix_lig.view(-1, 3, 3)
        # out_6d_rec = compute_rotation_matrix_from_ortho6d(rotation_matrix_rec)
        # out_6d_lig = compute_rotation_matrix_from_ortho6d(rotation_matrix_lig)
        
        translation_vector_rec = self.fc_translation_rec(attn_output_rec)
        translation_vector_lig = self.fc_translation_lig(attn_output_lig)
        
#         # Transform receptor coordinates
        # rotation_rec = rotation_matrix_rec[batch['receptor'].batch]
        rotation_rec = rotation_matrix_rec# Select rotation per batch
        translation_rec = translation_vector_rec # Select translation per batch
        receptor_coords = torch.einsum('ijk,ik->ij', rotation_rec, batch['receptor'].pos) + translation_rec

#         # Transform ligand coordinates
        rotation_lig = rotation_matrix_lig   # Select rotation per batch
        translation_lig = translation_vector_lig  # Select translation per batch
        ligand_coords = torch.einsum('ijk,ik->ij', rotation_lig, batch['ligand'].pos) + translation_lig


        return receptor_coords, ligand_coords


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
        #  print(receptor_coords)
        #  print(ligand_coords)
