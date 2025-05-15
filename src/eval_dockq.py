import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from biotite.structure.atoms import AtomArray, AtomArrayStack, stack
from torch_scatter import scatter
from torch.nn import Module
from pinder.core.utils.timer import timeit
from pinder.core.utils.process import process_starmap
from pinder.core.structure import surgery
from pinder.core.structure.atoms import (
    apply_mask,
    assign_receptor_ligand,
    atom_array_from_pdb_file,
    get_seq_aligned_structures,
    get_per_chain_seq_alignments,
    invert_chain_seq_map,
    standardize_atom_array,
)
from pinder.core.structure.models import BackboneDefinition, ChainConfig
from pinder.eval.dockq import metrics
import time
import json 
# import gradio as gr
# from gradio_molecule3d import Molecule3D
import torch
from pinder.core import get_pinder_location
get_pinder_location()
from pytorch_lightning import LightningModule

import torch
import lightning.pytorch as pl
import torch.nn.functional as F

import torch.nn as nn
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_scatter import scatter
from torch.nn import Module
from pinder.core.structure.contacts import pairwise_contacts

import pinder.core as pinder
pinder.__version__
from torch_geometric.loader import DataLoader
from pinder.core.loader.dataset import get_geo_loader
from pinder.core import download_dataset
from pinder.core import get_index
from pinder.core import get_metadata
from pathlib import Path
import pandas as pd
from pinder.core import PinderSystem
import torch
from pinder.core.loader.dataset import PPIDataset
from pinder.core.loader.geodata import NodeRepresentation
import pickle
from pinder.core import get_index, PinderSystem
from torch_geometric.data import HeteroData
import os

from enum import Enum

import numpy as np
import torch
import lightning.pytorch as pl
from numpy.typing import NDArray
from torch_geometric.data import HeteroData

from pinder.core.index.system import PinderSystem
from pinder.core.loader.structure import Structure
from pinder.core.utils import constants as pc
from pinder.core.utils.log import setup_logger
from pinder.core.index.system import _align_monomers_with_mask
from pinder.core.loader.structure import Structure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_scatter import scatter
from torch.nn import Module
import time
from torch_geometric.nn import global_max_pool
import copy
import inspect
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader import (
    LinkLoader,
    LinkNeighborLoader,
    NeighborLoader,
    NodeLoader,
)
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.loader.utils import get_edge_label_index, get_input_nodes
from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.typing import InputEdges, InputNodes

try:
    from lightning.pytorch import LightningDataModule as PLLightningDataModule
    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    PLLightningDataModule = object
    no_pytorch_lightning = True

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch_geometric.data.lightning.datamodule import LightningDataset
from pytorch_lightning.loggers.wandb import WandbLogger
def get_system(system_id: str) -> PinderSystem:
    return PinderSystem(system_id)
# from Bio import PDB
# from Bio.PDB.PDBIO import PDBIO
from biotite.structure.atoms import coord as to_coord
from pinder.core.structure.atoms import atom_array_from_pdb_file
from pathlib import Path
from pinder.eval.dockq.biotite_dockq import  DecoyDockQ, get_irmsd_interface
import biotite.structure as struc
from typing import List, NamedTuple, Set, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
from pinder.core.structure.atoms import (
    atom_array_from_pdb_file,
    atom_vdw_radius,
    filter_atoms,
)
from pinder.core.loader.geodata import structure2tensor
from models.pinder_module import PinderLitModule
from fastpdb import RustPDBFile
from multiprocessing import freeze_support
_Contacts = Set[Tuple[str, str, int, int]]
_StackContacts = List[_Contacts]
_AtomResContacts = Union[Tuple[_Contacts, _Contacts], _Contacts]
_StackAtomResContacts = Union[Tuple[_StackContacts, _StackContacts], _StackContacts]
class ContactPairs(NamedTuple):
    residue_contacts: _Contacts
    atom_contacts: _Contacts
from loguru import logger
try:
    from torch_cluster import knn_graph

    torch_cluster_installed = True
except ImportError:
    logger.warning(
        "torch-cluster is not installed!"
        "Please install the appropriate library for your pytorch installation."
        "See https://github.com/rusty1s/pytorch_cluster/issues/185 for background."
    )
    torch_cluster_installed = False

import rootutils
    
# setup root dir and pythonpath
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    
model = PinderLitModule.load_from_checkpoint("/home/sukanya/pinder_challenge/pinder_challenge-1/logs/train/runs/2024-12-26_07-36-50/checkpoints/last.ckpt")
trainer = pl.Trainer(
   
   
   fast_dev_run=False,
   accelerator="gpu" if torch.cuda.is_available() else "cpu",
   precision="bf16-mixed",
  
   devices=1,
)
model.eval()
log = setup_logger(__name__)

class BiotiteDockQ:
    """Biotite interface for fast calculation of DockQ and CAPRI metrics.

    Takes as input one native and all it's decoys
    from arbitrary number of methods to compare.
    """

    def __init__(
        self,
        native: Path,
        decoys: list[Path] | Path,
        native_receptor_chain: list[str] | None = None,
        native_ligand_chain: list[str] | None = None,
        decoy_receptor_chain: list[str] | None = None,
        decoy_ligand_chain: list[str] | None = None,
        pdb_engine: str = "fastpdb",
        backbone_definition: BackboneDefinition = "dockq",
        parallel_io: bool = True,
        max_workers: int | None = None,
    ) -> None:
        if not isinstance(decoys, list):
            decoys = [decoys]
            
        self.native_pdb = native
        self.decoy_pdbs = decoys
        self.pdb_engine = pdb_engine
        self.backbone_definition = backbone_definition
        self.parallel_io = parallel_io
        self.max_workers = max_workers
        self.native = atom_array_from_pdb_file(self.native_pdb)
        
        self.raw_native_shape = self.native.shape[0]
        # Im puzzled here, since the default DockQ keeps heavy atoms
        # but does nothing about the decoys missing the heavy atoms
        # This would penalize decoys without hydrogens when calculating fnat
        # and iRMSD
        # self.native = filter_atoms(native, calpha_only=False, backbone_only=False, heavy_only=True)
        if not (native_receptor_chain and native_ligand_chain):
            native_receptor_chain, native_ligand_chain = assign_receptor_ligand(
                self.native, set(self.native.chain_id)
            )
        self.native_rec_chain = native_receptor_chain
        self.native_lig_chain = native_ligand_chain
        self.model_rec_chain = decoy_receptor_chain
        self.model_lig_chain = decoy_ligand_chain



    def calculate(self) -> pd.DataFrame:
        self.prepare_inputs()
        if isinstance(self.decoy_stack, AtomArrayStack):
            ddq = DecoyDockQ(
                native=self.native,
                decoy=self.decoy_stack,
                chain_config=self.chain_config,
                backbone_definition=self.backbone_definition,
                native_contacts=self.native_contacts,
                native_interface=self.native_interface,
            )
            dockq_metrics = ddq.get_metrics()
            assert isinstance(dockq_metrics, pd.DataFrame)
            final_decoy_shapes = [arr.shape[0] for arr in ddq.decoy_stack]
            dockq_metrics.loc[:, "initial_decoy_shape"] = self.raw_decoy_shapes
            dockq_metrics.loc[:, "final_decoy_shape"] = final_decoy_shapes
            dockq_metrics.loc[:, "initial_native_shape"] = self.raw_native_shape
            dockq_metrics.loc[:, "final_native_shape"] = ddq.native.shape[0]
        else:
            dockq_metrics = []
            for i, pose in enumerate(self.decoy_stack):
                ddq = DecoyDockQ(
                    native=self.native,
                    decoy=pose,
                    chain_config=self.chain_config,
                    backbone_definition=self.backbone_definition,
                    native_contacts=self.native_contacts,
                    native_interface=self.native_interface,
                )
                pose_metrics = ddq.get_metrics()
                pose_metrics["initial_decoy_shape"] = self.raw_decoy_shapes[i]
                pose_metrics["final_decoy_shape"] = ddq.decoy_stack.shape[0]
                pose_metrics["initial_native_shape"] = self.raw_native_shape
                pose_metrics["final_native_shape"] = ddq.native.shape[0]
                dockq_metrics.append(pose_metrics)
            dockq_metrics = pd.DataFrame(dockq_metrics)

        assert isinstance(dockq_metrics, pd.DataFrame)
        # dockq_metrics.loc[:, "model_name"] = [decoy.stem for decoy in self.decoy_pdbs]
        # dockq_metrics.loc[:, "native_name"] = self.native_pdb.stem
        # dockq_metrics.loc[:, "system"] = self.native_pdb.stem
        # dockq_metrics.loc[:, "method"] = self.native_pdb.parent.parent.stem
        # dockq_metrics.loc[:, "model_folder"] = self.decoy_pdbs[0].parent.stem
        # col_order = [
        #     "model_name",
        #     "native_name",
        #     "system",
        #     "method",
        #     "model_folder",
        #     "iRMS",
        #     "LRMS",
        #     "Fnat",
        #     "DockQ",
        #     "CAPRI",
        #     "decoy_contacts",
        #     "native_contacts",
        #     "initial_decoy_shape",
        #     "final_decoy_shape",
        #     "initial_native_shape",
        #     "final_native_shape",
        # ]
        # self.metrics = dockq_metrics[col_order].copy()

        return dockq_metrics





    def get_native_contacts(self) -> None:
        # Used to define interface for iRMSD
        assert isinstance(self.native_rec_chain, list)
        assert isinstance(self.native_lig_chain, list)
        
        self.native_interface = get_irmsd_interface(
            self.native,
            self.native_rec_chain,
            self.native_lig_chain,
            backbone_only=False,
        )
        # Used to define Fnat
        self.native_contacts = pairwise_contacts(
            self.native,
            self.native_rec_chain,
            self.native_lig_chain,
        )
        # print(self.native_contacts , "Native contacts")




    def prepare_inputs(self) -> None:
        arr_list, decoy_pdbs = self.read_decoys(
            self.decoy_pdbs,
            self.pdb_engine,
            self.parallel_io,
            self.max_workers,
        )
        # print(arr_list , "dd")
        self.decoy_pdbs = decoy_pdbs
        if not (self.model_rec_chain and self.model_lig_chain):
            # Determine chain based on atom count
            self.model_rec_chain, self.model_lig_chain = assign_receptor_ligand(
                arr_list[0], set(arr_list[0].chain_id)
            )

        assert isinstance(self.model_rec_chain, list)
        assert isinstance(self.model_lig_chain, list)
        assert isinstance(self.native_rec_chain, list)
        assert isinstance(self.native_lig_chain, list)

        chain_remap = {
            existing: new
            for existing, new in zip(self.model_rec_chain, self.native_rec_chain)
        }
        for existing, new in zip(self.model_lig_chain, self.native_lig_chain):
            chain_remap[existing] = new

        # Re-name chains to match native.
        # filter_intersect causes differing chains to cause annotation mismatch
        self.chain_config = ChainConfig(
            decoy_receptor=self.native_rec_chain,
            decoy_ligand=self.native_lig_chain,
            native_receptor=self.native_rec_chain,
            native_ligand=self.native_lig_chain,
        )
        all_chains = self.native_rec_chain + self.native_lig_chain
        self.native = self.native[np.isin(self.native.chain_id, all_chains)]
        log.debug(f"Will use the following chain pairings:\n{self.chain_config}")

        self.raw_decoy_shapes = [arr.shape[0] for arr in arr_list]
        self.decoy_stack, self.native = self.create_decoy_stack(
            arr_list,
            self.native,
            R_chain=self.model_rec_chain,
            L_chain=self.model_lig_chain,
        )
        # Contacts need to be calculated here in case native renumbered above
        self.get_native_contacts()
        if isinstance(self.decoy_stack, AtomArrayStack):
            self.decoy_stack.chain_id = np.array(
                [chain_remap.get(ch, ch) for ch in self.decoy_stack.chain_id]
            )
            chain_mask = np.isin(self.decoy_stack.chain_id, all_chains)
            self.decoy_stack = apply_mask(self.decoy_stack, chain_mask)
        else:
            for i, arr in enumerate(self.decoy_stack):
                arr.chain_id = np.array(
                    [chain_remap.get(ch, ch) for ch in arr.chain_id]
                )
                arr = apply_mask(arr, np.isin(arr.chain_id, all_chains))
                self.decoy_stack[i] = arr.copy()
        self.set_common()




    @staticmethod
    @timeit
    def read_decoys(
        decoys: list[Path],
        pdb_engine: str,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> tuple[list[AtomArray], list[Path]]:
        # print(decoys,"Krupa")
        from itertools import repeat

        arr_list = process_starmap(
            atom_array_from_pdb_file,
            zip(decoys, repeat(pdb_engine)),
            parallel=parallel,
            max_workers=max_workers,
        )
        # print(arr_list,"Mauli")
        valid_arr = []
        valid_decoys = []
        for arr, decoy in zip(arr_list, decoys):
            if arr is not None:
                # print(set(arr.chain_id),"Mauli")
                if len(set(arr.chain_id)) > 1:
                    valid_arr.append(arr)
                    valid_decoys.append(decoy)

        arr_ordered = [(i, arr) for i, arr in enumerate(valid_arr)]
        arr_ordered = sorted(arr_ordered, key=lambda x: x[1].shape[0])
        
        ordered_decoys = []
        ordered_arrays = []
        for i, arr in arr_ordered:
            ordered_decoys.append(valid_decoys[i])
            ordered_arrays.append(arr)
        return ordered_arrays, ordered_decoys





    @staticmethod
    def create_decoy_stack(
        arr_list: list[AtomArray],
        native: AtomArray,
        R_chain: list[str],
        L_chain: list[str],
    ) -> tuple[AtomArrayStack, AtomArray]:
        try:
            # All annotations are equal and atom counts identical
            # print(arr_list)
            decoy_stack = stack(arr_list)
        except Exception:
            try:
                log.info(
                    "Couldnt create stack. Attempting chain and atom re-ordering..."
                )
                # First check if standardizing chain + atom order can make annotation arrays equal
                for i, arr in enumerate(arr_list):
                    arr_R = arr[np.isin(arr.chain_id, R_chain)].copy()
                    arr_L = arr[np.isin(arr.chain_id, L_chain)].copy()
                    arr_RL = arr_R + arr_L
                    arr_ordered = standardize_atom_array(arr_RL)
                    arr_list[i] = arr_ordered.copy()
                decoy_stack = stack(arr_list)
                log.info("Successfully created stack after standardizing order")
            except Exception as e:
                log.warning(
                    "Models have unequal annotations and/or shapes, not using vectorized AtomArrayStack"
                )
                decoy_stack = arr_list
                # print(arr_list, "models")
                # # Requires slower curation of intersecting atoms before stacking
                # decoy_stack, standardize_order = stack_filter_intersect(arr_list)
                # nat = native.copy()
                # if hasattr(nat, "atom_id"):
                #     nat.set_annotation("atom_id", np.repeat(0, nat.shape[0]))
                # native = standardize_atom_array(nat, standardize_order)
        return decoy_stack, native





    @timeit
    def set_common(self) -> None:
        # Ensure chain order is same between native and decoys
        # This still doesn't guarantee chain order is same for multi-chain case
        self.native = surgery.set_canonical_chain_order(
            self.native, self.chain_config, "native"
        )
        self.decoy_stack = surgery.set_canonical_chain_order(
            self.decoy_stack, self.chain_config, "decoy"
        )
 
def get_props_pdb(pdb_file):
    structure = Structure.read_pdb(pdb_file)
    atom_mask = np.isin(getattr(structure, "atom_name"), list(["CA"]))
    calpha = structure[atom_mask].copy()
    props = structure2tensor(
        atom_coordinates=structure.coord,
        atom_types=structure.atom_name,
        element_types=structure.element,
        residue_coordinates=calpha.coord,
        residue_types=calpha.res_name,
        residue_ids=calpha.res_id,
    )
    return structure, props       
        
def update_pdb_coordinates_from_tensor_list(n,input_filename_list, output_filename_parent, coordinates_tensor):
    r"""
    Updates atom coordinates in a PDB file with new transformed coordinates provided in a tensor.

    Parameters:
    - input_filename (str): Path to the original PDB file.
    - output_filename (str): Path to the new PDB file to save updated coordinates.
    - coordinates_tensor (torch.Tensor): Tensor of shape (1, N, 3) with transformed coordinates.
    """
    # Convert the tensor to a list of tuples
    new_coordinates = coordinates_tensor
    # print(new_coordinates , new_coordinates.shape,'lala')
    # Create a parser and parse the structure
    # parser = PDB.PDBParser(QUIET=True)
    # output_list=[]
    i = 0
    # n = 0
    for input_filename  in input_filename_list:
        # print(input_filename)
        input_filename = f"/home/sukanya/pinder_challenge/pinder_challenge-1/data/raw/apo/test/{input_filename}"
        structure , props = get_props_pdb(input_filename)
        # print(coords, coords.shape ,'and', structure.coord , structure.coord.shape)
        # (print(structure.coord.shape , new_coordinates.shape ))
        pdb_name = Path(input_filename).stem
        structure.coord = new_coordinates.detach().cpu().numpy()
        out_struct = Structure(
                                filepath=Path(f"./{output_filename_parent}_{pdb_name}_{n}.pdb"), atom_array=structure
                                )
        
        out_struct.to_pdb()
        out_pdb = f"./{output_filename_parent}_{pdb_name}_{n}.pdb"
        
        

    return out_pdb


 
# def merge_pdb_files(file1_list, file2_list, output_file_parent):
#     """
#     Merges two PDB files by concatenating them, with a specified chain ID in the second file replaced
#     to avoid conflicts. Skips specific lines like 'CRYST1', 'TER', and 'END'.
    
#     Parameters:
#     - file1 (str): Path to the first PDB file (e.g., receptor).
#     - file2 (str): Path to the second PDB file (e.g., ligand).
#     - output_file (str): Path to the output file where the merged structure will be saved.
#     - chain_to_replace (str): Chain ID in `file2` to replace (e.g., "A").
#     - new_chain (str): New chain ID to use in `file2` (e.g., "B").

#     Returns:
#     - str: Path to the merged output file.
#     """
#     output_file_list = []
#     i=0
    
#     for file2 in file2_list:
#         output_file = f"{output_file_parent}_{i}.pdb"
#         with open(output_file, 'w') as outfile:
#         # Copy contents from the first file, skipping 'CRYST1', 'TER', and 'END' lines
#              with open(file2, 'r') as f2:
#                   lines = f2.readlines()
#                 #   print(lines)
#                   for line in lines:
#                       if not line.startswith(('CRYST1', 'TER', 'END')):
#                          outfile.write(line)

#         # Copy contents from the second file, replacing the chain ID if it matches `chain_to_replace`
#              with open(file1_list[i], 'r') as f1:
#                   lines = f1.readlines()
                 
#                   for line in lines:
#                       if not line.startswith(('CRYST1', 'TER')):
#                     # Replace the chain ID at position 21 if it matches `chain_to_replace`
                         
                            
#                             outfile.write(line)
#         output_file_list.append(output_file)
#         i +=1
#     # print(f"Merged PDB saved to {output_file}")
#     return output_file_list
 
def calculate_dockq_metrics(natives, out_pdbs):
    # ligand_pdbs = update_pdb_coordinates_from_tensor_list(data["ligand"].path, "holo_ligand", ligands)
    # receptor_pdbs = update_pdb_coordinates_from_tensor_list(data["receptor"].path, "holo_receptor", receptors)
    # out_pdbs = merge_pdb_files(ligand_pdbs, receptor_pdbs, "output_orig")
    
    for native, decoy in zip(natives , out_pdbs):
            bdq = BiotiteDockQ(
                  native=native,
                  decoys=decoy,
                  native_receptor_chain=["R"],
                  native_ligand_chain=["L"],
                  decoy_receptor_chain=["R"],
                  decoy_ligand_chain=["L"],
                  parallel_io=True,
                  max_workers=2
                  )
            dockq = bdq.calculate()
            
            # for ligand, receptor , output in zip(ligand_pdbs, receptor_pdbs,out_pdbs):
            #     os.remove(ligand)
            #     os.remove(receptor)
            #     os.remove(output)
    # DockQ_df = pd.concat(dockQ, ignore_index=True)
    return dockq  
 


datamodule = "/home/sukanya/pinder_challenge/pinder_challenge-1/data/raw/apo/test"
def create_graph(pdb, k=5, device: torch.device = torch.device("cpu")):

    structure, props = get_props_pdb(pdb)
    
    # receptor_structure, props_receptor = get_props_pdb(pdb_2)

    data = HeteroData()
    
    for chain in ['L', 'R']:
        # Get indices where chain_id matches the current chain
        chain_indices = np.where(structure.chain_id == chain)[0]

        if len(chain_indices) > 0:  # Proceed only if this chain exists
            filtered_atom_types = props["atom_types"][chain_indices]
            filtered_atom_coordinates = props["atom_coordinates"][chain_indices]

            if chain == 'L':  # For ligand
                data["ligand"].x = filtered_atom_types
                data["ligand"].pos = filtered_atom_coordinates
                data["ligand", "ligand"].edge_index = knn_graph(data["ligand"].pos, k=k)
            elif chain == 'R':  # For receptor
                data["receptor"].x = filtered_atom_types
                data["receptor"].pos = filtered_atom_coordinates
                data["receptor", "receptor"].edge_index = knn_graph(data["receptor"].pos, k=k)

    data = data.to(device)
    return data, structure

apo_pdbs = "/home/sukanya/pinder_challenge/pinder_challenge-1/data/raw/apo/test"
holo_pdbs = "/home/sukanya/pinder_challenge/pinder_challenge-1/data/raw/holo/test"
def apply_model(model, input_tensor):
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_tensor)
    return output
import os
import torch
if __name__ == '__main__':
    freeze_support()
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    torch.cuda.empty_cache()
    # Initialize output lists and counter
    Out = []
    dockq = []
    
    i = 321
    output_parent = "/home/sukanya/pinder_challenge/pinder_challenge-1/data/output"
    pdb_directory = "/home/sukanya/pinder_challenge/pinder_challenge-1/data/output/saved"

    # Ensure output directories exist
    os.makedirs(output_parent, exist_ok=True)
    os.makedirs(pdb_directory, exist_ok=True)

    # Iterate over the files
    for file_name, apo_complex, holo_complex in list(zip(os.listdir(datamodule), os.listdir(apo_pdbs), os.listdir(holo_pdbs)))[338:]:
        file_path = os.path.join(datamodule, file_name)
        
        # Read holo PDB file
        holo_pdb_file = os.path.join("/home/sukanya/pinder_challenge/pinder_challenge-1/data/raw/holo/test", holo_complex)
    
        data , _ = create_graph(file_path)
        # print('data', data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        try:
            # Process the data with the model
            receptor_coords, ligand_coords = model(data.to(device))
            # print('coord',receptor_coords.shape,ligand_coords.shape  )
        except Exception as e:
            print(f"Error during model inference on {file_name}: {e}")
            continue

        # Concatenate receptor and ligand coordinates
        out_coords = torch.cat((receptor_coords, ligand_coords), dim=0)
        Out.append(out_coords)

        # Update PDB file with new coordinates
        new_pdb = update_pdb_coordinates_from_tensor_list(i,[apo_complex], "updated_apo_complex", out_coords)
        print(new_pdb)
        # print(holo_pdb_file,"s",new_pdb,"k")
        # Calculate DockQ metric
        bdq = BiotiteDockQ(
            native=holo_pdb_file,
            decoys=new_pdb,
            native_receptor_chain=["R"],
            native_ligand_chain=["L"],
            decoy_receptor_chain=["R"],
            decoy_ligand_chain=["L"],
            parallel_io=True,
            max_workers=2
        )
        
        df = bdq.calculate()
        
        df.to_csv(f"dockq_{i}.csv", index = False)
        dockq.append(df)
        i += 1
        
        
        
    # Concatenate results into a DataFrame
    DockQ_df = pd.concat(dockq, ignore_index=True)
    
    DockQ_df.to_csv('DockQ_df_apo_1.csv', index=False)
    
# # Create the pie chart for the 'CAPRI' column
#     capri_counts = DockQ_df['CAPRI'].value_counts()

# # Plot and save the pie chart
#     plt.figure(figsize=(6, 6))
#     capri_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3')
#     plt.title('Distribution of CAPRI Classes')
#     plt.ylabel('')  # Hide y-label for aesthetics
#     plt.savefig('capri_pie_chart.jpg', format='jpg')
#     plt.close()

# # Plot distributions for 'irms', 'fnat', and 'lrms'
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # 'irms' distribution
#     axes[0, 0].plot(DockQ_df['iRMS'], marker='o', linestyle='-', color='skyblue')
#     axes[0, 0].set_title('iRMS')
#     axes[0, 0].set_xlabel('Sample Index')
#     axes[0, 0].set_ylabel('iRMS')

# # Plot LRMS
#     axes[0, 1].plot(DockQ_df['LRMS'], marker='o', linestyle='-', color='lightgreen')
#     axes[0, 1].set_title('LRMS')
#     axes[0, 1].set_xlabel('Sample Index')
#     axes[0, 1].set_ylabel('LRMS')

# # Plot Fnat
#     axes[1, 0].plot(DockQ_df['Fnat'], marker='o', linestyle='-', color='lightcoral')
#     axes[1, 0].set_title('Fnat')
#     axes[1, 0].set_xlabel('Sample Index')
#     axes[1, 0].set_ylabel('Fnat')

# # Plot DockQ
#     axes[1, 1].plot(DockQ_df['DockQ'], marker='o', linestyle='-', color='lightskyblue')
#     axes[1, 1].set_title('DockQ')
#     axes[1, 1].set_xlabel('Sample Index')
#     axes[1, 1].set_ylabel('DockQ')

# # Adjust layout and save the figure
#     plt.tight_layout()
#     plt.savefig('iRMS_LRMS_Fnat_DockQ_plots.jpg', format='jpg')
#     plt.show()